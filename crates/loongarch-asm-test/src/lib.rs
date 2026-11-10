//! Assembly-level checks for the LoongArch64 LSX/LASX intrinsics.
//!
//! The list of intrinsics and their signatures is derived from the specs
//! that drive `stdarch-gen-loongarch` (`crates/stdarch-gen-loongarch/{lsx,lasx}.spec`),
//! which cover both the LLVM-intrinsic based and the portable (`intrinsics::simd`)
//! implementations in `core_arch`.
//!
//! For each intrinsic a tiny wrapper function is generated:
//!
//! ```ignore
//! #[unsafe(no_mangle)]
//! #[target_feature(enable = "lsx")]
//! pub unsafe fn lsx_vadd_b() {
//!     let a: m128i;
//!     let b: m128i;
//!     asm!("# in: a={0}, b={1}", out(vreg) a, out(vreg) b, ...);
//!     let r = arch::lsx_vadd_b(a, b);
//!     asm!("# out: {0}", in(vreg) r, ...);
//! }
//! ```
//!
//! The empty `asm!` blocks hand the operands over in registers, so the values are
//! opaque to the optimizer (no constant folding) and there is no ABI noise (no
//! loads/stores for by-reference vector arguments, no sign extension of return
//! values). The generated body is then just the instruction(s) the intrinsic lowers to.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

/// The LoongArch SIMD extension an intrinsic belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ext {
    Lsx,
    Lasx,
}

impl Ext {
    pub const ALL: [Ext; 2] = [Ext::Lsx, Ext::Lasx];

    pub fn name(self) -> &'static str {
        match self {
            Ext::Lsx => "lsx",
            Ext::Lasx => "lasx",
        }
    }
}

/// Rust-level types appearing in intrinsic signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    M128i,
    M128,
    M128d,
    M256i,
    M256,
    M256d,
    I32,
    I64,
    U32,
    U64,
    ConstPtr,
    MutPtr,
}

impl Ty {
    fn from_spec(t: &str, is_store: bool) -> Ty {
        match t {
            "V16QI" | "V8HI" | "V4SI" | "V2DI" | "UV16QI" | "UV8HI" | "UV4SI" | "UV2DI" => {
                Ty::M128i
            }
            "V32QI" | "V16HI" | "V8SI" | "V4DI" | "UV32QI" | "UV16HI" | "UV8SI" | "UV4DI" => {
                Ty::M256i
            }
            "V4SF" => Ty::M128,
            "V8SF" => Ty::M256,
            "V2DF" => Ty::M128d,
            "V4DF" => Ty::M256d,
            "SI" | "QI" | "HI" => Ty::I32,
            "DI" => Ty::I64,
            "USI" | "UQI" => Ty::U32,
            "UDI" => Ty::U64,
            "CVPOINTER" if is_store => Ty::MutPtr,
            "CVPOINTER" => Ty::ConstPtr,
            _ => panic!("unknown spec data type `{t}`"),
        }
    }

    pub fn rust(self) -> &'static str {
        match self {
            Ty::M128i => "m128i",
            Ty::M128 => "m128",
            Ty::M128d => "m128d",
            Ty::M256i => "m256i",
            Ty::M256 => "m256",
            Ty::M256d => "m256d",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::ConstPtr => "*const i8",
            Ty::MutPtr => "*mut i8",
        }
    }

    /// Type as written in the generated wrapper crate.
    fn wrapper(self) -> String {
        match self {
            Ty::M128i | Ty::M128 | Ty::M128d | Ty::M256i | Ty::M256 | Ty::M256d => {
                format!("arch::{}", self.rust())
            }
            _ => self.rust().to_string(),
        }
    }

    /// `asm!` register class used to pass a value of this type.
    fn reg_class(self) -> &'static str {
        match self {
            Ty::M128i | Ty::M128 | Ty::M128d => "vreg",
            Ty::M256i | Ty::M256 | Ty::M256d => "xreg",
            _ => "reg",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: &'static str,
    pub ty: Ty,
}

/// A const generic (immediate) parameter.
#[derive(Clone, Debug)]
pub struct ConstParam {
    pub name: String,
    pub signed: bool,
    pub bits: u32,
    /// The value used in the generated test.
    pub value: i64,
}

impl ConstParam {
    fn new(name: String, signed: bool, bits: u32, value: i64) -> Self {
        let (lo, hi) = if signed {
            (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
        } else {
            (0, (1i64 << bits) - 1)
        };
        assert!(
            (lo..=hi).contains(&value),
            "immediate {value} for {name} out of range {lo}..={hi}"
        );
        ConstParam {
            name,
            signed,
            bits,
            value,
        }
    }

    fn rust_ty(&self) -> &'static str {
        if self.signed { "i32" } else { "u32" }
    }
}

#[derive(Clone, Debug)]
pub struct Intrinsic {
    pub name: String,
    pub ext: Ext,
    /// Implemented with `intrinsics::simd` (portable) rather than an LLVM intrinsic.
    pub portable: bool,
    pub asm_fmts: Vec<String>,
    pub data_types: Vec<String>,
    pub consts: Vec<ConstParam>,
    pub params: Vec<Param>,
    pub ret: Option<Ty>,
    pub is_unsafe: bool,
    pub target_feature: String,
}

/// Immediate used for memory offsets: non-zero, and a multiple of every element size so
/// it is valid for the scaled offsets of `vldrepl.{h,w,d}` and `vstelm.{h,w,d}`.
const MEM_OFFSET: i64 = 8;

/// Default immediates for everything else: a mid-range value, `2^(bits-1) - 1` for
/// unsigned (but at least 1) and `5` for signed immediates.
///
/// Small values like `0`, `1` or `-1` are avoided on purpose: LLVM rewrites operations
/// with such constants into other instructions (`vmini.hu x, 1` becomes `vseqi.h` +
/// `vandn.v`, `vori.b x, 1` becomes `vbitseti.b`, `vrepli.h -1` becomes `vrepli.b -1`,
/// shifts by zero vanish, ...), which would hide what the intrinsic itself lowers to.
fn default_imm(signed: bool, bits: u32) -> i64 {
    if signed {
        5
    } else {
        ((1i64 << (bits - 1)) - 1).max(1)
    }
}

/// Per-intrinsic immediates, overriding the defaults above (in const-generic order).
const IMM_OVERRIDES: &[(&str, &[i64])] = &[
    // Mode 0b0001 (`imm8 << 8` in each word, here 0x0000ab00): only encodable with `vldi`,
    // not with `vrepli.[bhwd]`.
    ("lsx_vldi", &[0x11ab - 0x2000]),
    ("lasx_xvldi", &[0x11ab - 0x2000]),
    // `x & 0x7f` would be turned into `vbitclri.b x, 7`.
    ("lsx_vandi_b", &[0x55]),
    ("lasx_xvandi_b", &[0x55]),
    // Lanes (1, 2): with 127 both lanes pick the same element, which becomes `vreplvei.d`.
    ("lsx_vshuf4i_d", &[0b1001]),
    ("lasx_xvshuf4i_d", &[0b1001]),
    // Select different 128-bit halves; avoid bits 3 and 7, which zero a half.
    ("lasx_xvpermi_q", &[0x31]),
];

fn imm_value(name: &str, idx: usize, is_mem_offset: bool, signed: bool, bits: u32) -> i64 {
    if let Some((_, v)) = IMM_OVERRIDES.iter().find(|(n, _)| *n == name) {
        return v[idx];
    }
    if is_mem_offset {
        MEM_OFFSET
    } else {
        default_imm(signed, bits)
    }
}

fn target_feature(ext: Ext, name: &str) -> String {
    let frecipe = matches!(
        name,
        "lsx_vfrecipe_s"
            | "lsx_vfrecipe_d"
            | "lsx_vfrsqrte_s"
            | "lsx_vfrsqrte_d"
            | "lasx_xvfrecipe_s"
            | "lasx_xvfrecipe_d"
            | "lasx_xvfrsqrte_s"
            | "lasx_xvfrsqrte_d"
    );
    format!("{}{}", ext.name(), if frecipe { ",frecipe" } else { "" })
}

/// Parse the number of bits out of an immediate asm format like `ui5`, `si12` or `i13`.
fn imm_bits(fmt: &str) -> u32 {
    fmt.trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .parse()
        .unwrap_or_else(|_| panic!("bad immediate format `{fmt}`"))
}

impl Intrinsic {
    /// Build an intrinsic from a spec entry, deriving its Rust signature with the same
    /// rules `stdarch-gen-loongarch` uses.
    fn from_spec(
        ext: Ext,
        name: String,
        portable: bool,
        asm_fmts: Vec<String>,
        data_types: Vec<String>,
    ) -> Self {
        let is_store = name.contains("vst");
        let ty = |t: &str| Ty::from_spec(t, is_store);
        let out_t = data_types[0].as_str();
        let in_t: Vec<&str> = data_types[1..].iter().map(|s| s.as_str()).collect();
        let n = in_t.len();
        let fmt = |i: usize| asm_fmts[i].as_str();
        const NAMES: [&str; 4] = ["a", "b", "c", "d"];

        let mut consts = Vec::new();
        let mut add_const = |cname: String, signed: bool, bits: u32, mem: bool| {
            let idx = consts.len();
            let value = imm_value(&name, idx, mem, signed, bits);
            consts.push(ConstParam::new(cname, signed, bits, value));
        };

        let params: Vec<Param> = if n == 1 && in_t[0] == "HI" {
            // vldi / vrepli: the only operand is an immediate.
            let bits = imm_bits(fmt(1));
            add_const(format!("IMM_S{bits}"), true, bits, false);
            vec![]
        } else if n == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
            let bits = imm_bits(fmt(2));
            add_const(format!("IMM{bits}"), false, bits, false);
            vec![Param {
                name: "a",
                ty: ty(in_t[0]),
            }]
        } else if n == 2 && in_t[1] == "QI" {
            let bits = imm_bits(fmt(2));
            add_const(format!("IMM_S{bits}"), true, bits, false);
            vec![Param {
                name: "a",
                ty: ty(in_t[0]),
            }]
        } else if n == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
            let bits = imm_bits(fmt(2));
            add_const(format!("IMM_S{bits}"), true, bits, true);
            vec![Param {
                name: "mem_addr",
                ty: ty(in_t[0]),
            }]
        } else if n == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
            vec![
                Param {
                    name: "mem_addr",
                    ty: ty(in_t[0]),
                },
                Param {
                    name: "b",
                    ty: ty(in_t[1]),
                },
            ]
        } else if n == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
            let bits = imm_bits(fmt(2));
            add_const(format!("IMM{bits}"), false, bits, false);
            vec![
                Param {
                    name: "a",
                    ty: ty(in_t[0]),
                },
                Param {
                    name: "b",
                    ty: ty(in_t[1]),
                },
            ]
        } else if n == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
            let bits = imm_bits(fmt(2));
            add_const(format!("IMM_S{bits}"), true, bits, true);
            vec![
                Param {
                    name: "a",
                    ty: ty(in_t[0]),
                },
                Param {
                    name: "mem_addr",
                    ty: ty(in_t[1]),
                },
            ]
        } else if n == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
            vec![
                Param {
                    name: "a",
                    ty: ty(in_t[0]),
                },
                Param {
                    name: "mem_addr",
                    ty: ty(in_t[1]),
                },
                Param {
                    name: "b",
                    ty: ty(in_t[2]),
                },
            ]
        } else if n == 4 {
            // vstelm: offset + element index.
            let off_bits = imm_bits(fmt(2));
            let idx_bits = match name.chars().last().unwrap() {
                'b' => 4,
                'h' => 3,
                'w' => 2,
                'd' => 1,
                c => panic!("unexpected element suffix `{c}` in {name}"),
            };
            add_const(format!("IMM_S{off_bits}"), true, off_bits, true);
            add_const(format!("IMM{idx_bits}"), false, idx_bits, false);
            vec![
                Param {
                    name: "a",
                    ty: ty(in_t[0]),
                },
                Param {
                    name: "mem_addr",
                    ty: ty(in_t[1]),
                },
            ]
        } else {
            in_t.iter()
                .enumerate()
                .map(|(i, t)| Param {
                    name: NAMES[i],
                    ty: ty(t),
                })
                .collect()
        };

        let ret = if out_t.eq_ignore_ascii_case("void") {
            None
        } else {
            Some(ty(out_t))
        };
        let is_unsafe = in_t.contains(&"CVPOINTER");
        let target_feature = target_feature(ext, &name);
        Intrinsic {
            name,
            ext,
            portable,
            asm_fmts,
            data_types,
            consts,
            params,
            ret,
            is_unsafe,
            target_feature,
        }
    }

    /// The public Rust signature of the intrinsic, e.g.
    /// `pub fn lsx_vslli_b<const IMM3: u32>(a: m128i) -> m128i`.
    pub fn signature(&self) -> String {
        let mut s = format!(
            "pub {}fn {}",
            if self.is_unsafe { "unsafe " } else { "" },
            self.name
        );
        if !self.consts.is_empty() {
            let c: Vec<String> = self
                .consts
                .iter()
                .map(|c| format!("const {}: {}", c.name, c.rust_ty()))
                .collect();
            write!(s, "<{}>", c.join(", ")).unwrap();
        }
        let p: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.ty.rust()))
            .collect();
        write!(s, "({})", p.join(", ")).unwrap();
        if let Some(r) = self.ret {
            write!(s, " -> {}", r.rust()).unwrap();
        }
        s
    }

    /// The call expression used in the test, e.g. `lsx_vslli_b::<1>(a)`.
    pub fn call(&self, path: &str) -> String {
        let mut s = format!("{path}{}", self.name);
        if !self.consts.is_empty() {
            let c: Vec<String> = self.consts.iter().map(|c| c.value.to_string()).collect();
            write!(s, "::<{}>", c.join(", ")).unwrap();
        }
        let p: Vec<&str> = self.params.iter().map(|p| p.name).collect();
        write!(s, "({})", p.join(", ")).unwrap();
        s
    }

    /// What the generated code for this intrinsic is expected to look like.
    pub fn expect(&self) -> Expect {
        let base = self
            .name
            .strip_prefix("lsx_")
            .or_else(|| self.name.strip_prefix("lasx_"))
            .unwrap();
        let x = if self.ext == Ext::Lasx { "x" } else { "" };
        let v = |s: &[&str]| s.iter().map(|m| m.to_string()).collect::<Vec<_>>();
        let named = base.replace('_', ".");
        let mut e = Expect {
            named: vec![named.clone()],
            equivalent: vec![],
            allowed: vec![],
        };

        // `[x]b[n]z_*` lower to a `[x]vset*` into a condition flag and a branch on it.
        let bz = base.trim_start_matches('x');
        if let Some((op @ ("bz" | "bnz"), t)) = bz.split_once('_') {
            let m = match (op, t) {
                ("bz", "v") => "vseteqz.v".to_string(),
                ("bnz", "v") => "vsetnez.v".to_string(),
                ("bz", t) => format!("vsetanyeqz.{t}"),
                (_, t) => format!("vsetallnez.{t}"),
            };
            e.named = vec![format!("{x}{m}")];
            e.allowed = v(&["bcnez", "bceqz", "addi.w", "addi.d", "ori", "b", "movcf2gr"]);
            return e;
        }

        match base {
            // LASX helpers that have no instruction of their own: a 128-bit LSX register is
            // the low half of the corresponding 256-bit LASX register.
            b if b.starts_with("cast_128") || b.starts_with("extract_128_lo") => e.named = vec![],
            b if b.starts_with("extract_128_hi")
                || b.starts_with("concat_128")
                || b.starts_with("insert_128") =>
            {
                e.named = v(&["xvpermi.q"])
            }
            // The `_f` suffix only marks the float vector type.
            "xvpickve_w_f" | "xvpickve_d_f" => {
                e.named = vec![named.trim_end_matches(".f").to_string()]
            }
            // `vrepli.[bhwd]` are aliases of `vldi`.
            "vldi" | "xvldi" => {
                e.equivalent = ["b", "h", "w", "d"]
                    .iter()
                    .map(|t| format!("{x}vrepli.{t}"))
                    .collect()
            }
            b if b.starts_with("vrepli_") || b.starts_with("xvrepli_") => {
                e.equivalent = vec![format!("{x}vldi")]
            }
            // With two 64-bit lanes these all compute the same thing.
            "vpickev_d" | "vilvl_d" | "xvpickev_d" | "xvilvl_d" => {
                e.equivalent = vec![
                    format!("{x}vpackev.d"),
                    format!("{x}vpickev.d"),
                    format!("{x}vilvl.d"),
                ]
            }
            "vpickod_d" | "vilvh_d" | "xvpickod_d" | "xvilvh_d" => {
                e.equivalent = vec![
                    format!("{x}vpackod.d"),
                    format!("{x}vpickod.d"),
                    format!("{x}vilvh.d"),
                ]
            }
            // Splatting a 64-bit lane is the same as a 32-bit lane shuffle.
            "vreplvei_d" | "xvrepl128vei_d" => e.equivalent = vec![format!("{x}vshuf4i.w")],
            "xvreplve0_d" | "xvreplve0_q" => e.equivalent = v(&["xvpermi.d", "xvpermi.q"]),
            // On LA64 32-bit values live sign-extended in 64-bit registers, so extracting an
            // unsigned word is the same as a signed one; for doublewords there's no difference.
            "vpickve2gr_wu" | "xvpickve2gr_wu" | "vpickve2gr_du" | "xvpickve2gr_du" => {
                e.equivalent = vec![named.trim_end_matches('u').to_string()]
            }
            // Every `imm3` bit mask fits the 8-bit immediate of `v{and,or,xor}i.b`, so no
            // constant keeps LLVM from using those instead.
            "vbitclri_b" | "xvbitclri_b" => e.equivalent = vec![format!("{x}vandi.b")],
            "vbitseti_b" | "xvbitseti_b" => e.equivalent = vec![format!("{x}vori.b")],
            "vbitrevi_b" | "xvbitrevi_b" => e.equivalent = vec![format!("{x}vxori.b")],
            // `x <= imm` is canonicalized to `x < imm + 1` by LLVM.
            b if b.starts_with("vslei_") || b.starts_with("xvslei_") => {
                e.equivalent = vec![named.replacen("slei", "slti", 1)]
            }
            _ => {}
        }
        e
    }
}

/// The expected shape of the code generated for an intrinsic.
#[derive(Clone, Debug)]
pub struct Expect {
    /// The instruction the intrinsic corresponds to. Empty when it needs no instruction.
    pub named: Vec<String>,
    /// Other single instructions that compute the same result just as well.
    pub equivalent: Vec<String>,
    /// Instructions that may additionally appear (e.g. the branch of `bz`/`bnz`).
    pub allowed: Vec<String>,
}

/// Parse a `{lsx,lasx}.spec` file.
pub fn parse_spec(ext: Ext, text: &str) -> Vec<Intrinsic> {
    let mut out = Vec::new();
    let mut portable = false;
    let mut name: Option<String> = None;
    let mut asm_fmts = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line == "impl = portable" {
            portable = true;
        } else if let Some(n) = line.strip_prefix("name = ") {
            name = Some(n.trim().to_string());
        } else if let Some(f) = line.strip_prefix("asm-fmts = ") {
            asm_fmts = f.split(',').map(|v| v.trim().to_string()).collect();
        } else if let Some(d) = line.strip_prefix("data-types = ") {
            let data_types = d.split(',').map(|v| v.trim().to_string()).collect();
            let n = name.take().expect("data-types without name");
            out.push(Intrinsic::from_spec(
                ext,
                n,
                portable,
                std::mem::take(&mut asm_fmts),
                data_types,
            ));
            portable = false;
        }
    }
    out
}

/// Load all LSX and LASX intrinsics from the specs in `stdarch-gen-loongarch`.
pub fn load_intrinsics(repo_root: &Path) -> Vec<Intrinsic> {
    let mut all = Vec::new();
    for ext in Ext::ALL {
        let path = repo_root
            .join("crates/stdarch-gen-loongarch")
            .join(format!("{}.spec", ext.name()));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        all.extend(parse_spec(ext, &text));
    }
    all
}

const ASM_OPTS: &str = "options(nomem, nostack, preserves_flags)";

/// Generate the source of the wrapper crate.
pub fn gen_wrappers(intrinsics: &[Intrinsic]) -> String {
    let mut s = String::new();
    s.push_str(
        "// Generated by crates/loongarch-asm-test. DO NOT EDIT.\n\
         #![no_std]\n\
         #![feature(asm_experimental_reg)]\n\
         #![allow(unused_unsafe, non_snake_case, clippy::all)]\n\n\
         use core::arch::asm;\n\
         use core_arch::arch::loongarch64 as arch;\n",
    );
    for i in intrinsics {
        writeln!(s, "\n/// `{}`", i.signature()).unwrap();
        writeln!(s, "#[unsafe(no_mangle)]").unwrap();
        writeln!(s, "#[target_feature(enable = \"{}\")]", i.target_feature).unwrap();
        writeln!(s, "pub unsafe fn {}() {{", i.name).unwrap();
        writeln!(s, "    unsafe {{").unwrap();
        if !i.params.is_empty() {
            for p in &i.params {
                writeln!(s, "        let {}: {};", p.name, p.ty.wrapper()).unwrap();
            }
            let tmpl: Vec<String> = i
                .params
                .iter()
                .enumerate()
                .map(|(n, p)| format!("{}={{{n}}}", p.name))
                .collect();
            let ops: Vec<String> = i
                .params
                .iter()
                .map(|p| format!("out({}) {}", p.ty.reg_class(), p.name))
                .collect();
            writeln!(
                s,
                "        asm!(\"# in: {}\", {}, {ASM_OPTS});",
                tmpl.join(", "),
                ops.join(", ")
            )
            .unwrap();
        }
        match i.ret {
            Some(r) => {
                writeln!(s, "        let r: {} = {};", r.wrapper(), i.call("arch::")).unwrap();
                writeln!(
                    s,
                    "        asm!(\"# out: {{0}}\", in({}) r, {ASM_OPTS});",
                    r.reg_class()
                )
                .unwrap();
            }
            None => writeln!(s, "        {};", i.call("arch::")).unwrap(),
        }
        writeln!(s, "    }}\n}}").unwrap();
    }
    s
}

/// The cleaned-up assembly of one generated function.
#[derive(Clone, Debug, Default)]
pub struct FuncAsm {
    /// Lines for display (instructions, local labels and the operand comments).
    pub lines: Vec<String>,
    /// Instruction mnemonics, excluding `ret`.
    pub mnemonics: Vec<String>,
}

/// Extract the bodies of the functions named in `names` from `rustc --emit asm` output.
pub fn parse_asm(text: &str, names: &HashSet<&str>) -> HashMap<String, FuncAsm> {
    let mut out = HashMap::new();
    let mut cur: Option<(String, FuncAsm)> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(label) = line.strip_suffix(':')
            && names.contains(label)
        {
            cur = Some((label.to_string(), FuncAsm::default()));
            continue;
        }
        let Some((_, f)) = cur.as_mut() else { continue };
        if line.starts_with(".Lfunc_end") {
            let (name, f) = cur.take().unwrap();
            out.insert(name, f);
            continue;
        }
        if line.is_empty() || line == "#APP" || line == "#NO_APP" {
            continue;
        }
        if let Some(c) = line.strip_prefix('#') {
            let c = c.trim();
            if c.starts_with("in:") || c.starts_with("out:") {
                f.lines.push(format!(
                    "# {}",
                    c.split_whitespace().collect::<Vec<_>>().join(" ")
                ));
            }
            continue;
        }
        if line.starts_with('.') {
            if line.ends_with(':') && !line.starts_with(".Lfunc") && !line.starts_with(".Ltmp") {
                f.lines.push(line.to_string());
            }
            continue;
        }
        let insn = line.split(" #").next().unwrap().trim();
        let (mn, ops) = match insn.split_once(char::is_whitespace) {
            Some((m, o)) => (m, o.trim()),
            None => (insn, ""),
        };
        f.lines.push(if ops.is_empty() {
            mn.to_string()
        } else {
            format!("{mn:<16}{ops}")
        });
        if mn != "ret" {
            f.mnemonics.push(mn.to_string());
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Exactly the named instruction.
    Exact,
    /// A single instruction that is equivalent to the named one.
    Equivalent,
    /// The expected instruction plus other instructions.
    Extra,
    /// Neither the named instruction nor an equivalent one appears.
    Missing,
}

pub fn verdict(e: &Expect, f: &FuncAsm) -> Verdict {
    let has = |set: &[String], m: &String| set.iter().any(|x| x == m);
    let rest_allowed = |set: &[String]| {
        f.mnemonics
            .iter()
            .all(|m| has(set, m) || has(&e.allowed, m))
    };
    if e.named.is_empty() {
        return if f.mnemonics.is_empty() {
            Verdict::Exact
        } else {
            Verdict::Extra
        };
    }
    let named = f.mnemonics.iter().any(|m| has(&e.named, m));
    let equiv = f.mnemonics.iter().any(|m| has(&e.equivalent, m));
    match (named, equiv) {
        (true, _) if rest_allowed(&e.named) => Verdict::Exact,
        (false, true) if rest_allowed(&e.equivalent) => Verdict::Equivalent,
        (false, false) => Verdict::Missing,
        _ => Verdict::Extra,
    }
}

/// The parts of rustc's `--emit llvm-ir` output needed to build standalone modules.
#[derive(Debug, Default)]
pub struct LlvmModule {
    pub datalayout: String,
    pub triple: String,
    /// Function name -> lines of the body (between the `define` line and the closing `}`).
    pub functions: HashMap<String, Vec<String>>,
    /// Function name -> `declare` line.
    pub declares: HashMap<String, String>,
    /// Attribute group (`#N`) -> its contents.
    pub attributes: HashMap<String, String>,
    /// Function name -> attribute group (`#N`) of its definition.
    pub define_attrs: HashMap<String, String>,
}

/// Name of the function a `define`/`declare` line defines, e.g. `lsx_vadd_b`.
fn ir_fn_name(line: &str) -> Option<&str> {
    let at = line.find('@')?;
    let rest = &line[at + 1..];
    Some(&rest[..rest.find('(')?])
}

pub fn parse_llvm_ir(text: &str) -> LlvmModule {
    let mut m = LlvmModule::default();
    let mut cur: Option<(String, Vec<String>)> = None;
    for line in text.lines() {
        if let Some((name, body)) = cur.as_mut() {
            if line == "}" {
                m.functions
                    .insert(std::mem::take(name), std::mem::take(body));
                cur = None;
            } else {
                body.push(line.to_string());
            }
        } else if let Some(v) = line.strip_prefix("target datalayout = ") {
            m.datalayout = v.to_string();
        } else if let Some(v) = line.strip_prefix("target triple = ") {
            m.triple = v.to_string();
        } else if line.starts_with("define ") {
            let name = ir_fn_name(line).unwrap_or_default().to_string();
            if let Some(id) = line.split(' ').find(|w| w.starts_with('#')) {
                m.define_attrs.insert(name.clone(), id.to_string());
            }
            cur = Some((name, vec![]));
        } else if line.starts_with("declare ") {
            if let Some(name) = ir_fn_name(line) {
                m.declares.insert(name.to_string(), line.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("attributes ")
            && let Some((id, attrs)) = rest.split_once(" = ")
        {
            m.attributes.insert(id.to_string(), attrs.to_string());
        }
    }
    m
}

/// Rewrite every `%name` local value in `line` with `f` (when it returns `Some`).
fn map_locals(line: &str, f: impl Fn(&str) -> Option<String>) -> String {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(p) = rest.find('%') {
        out.push_str(&rest[..=p]);
        rest = &rest[p + 1..];
        let end = rest.find(|c: char| !is_ident(c)).unwrap_or(rest.len());
        let ident = &rest[..end];
        match f(ident) {
            Some(new) => out.push_str(&new),
            None => out.push_str(ident),
        }
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Split a leading IR type off `s`: `<2 x i64> %r` -> (`<2 x i64>`, `%r`).
fn split_ir_type(s: &str) -> (&str, &str) {
    let s = s.trim();
    let close = match s.chars().next() {
        Some('<') => '>',
        Some('{') => '}',
        _ => {
            let (t, v) = s.split_once(' ').unwrap_or((s, ""));
            return (t, v.trim());
        }
    };
    let open = s.chars().next().unwrap();
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return (&s[..=i], s[i + 1..].trim());
            }
        }
    }
    (s, "")
}

/// Remove metadata attachments (`, !srcloc !3`) and call-site attribute groups (`#15`).
fn strip_ir_annotations(line: &str) -> String {
    let mut s = line.to_string();
    while let Some(p) = s.find(", !") {
        let tail = &s[p + 2..];
        // `!kind !N`
        let mut it = tail.splitn(3, ' ');
        let (Some(kind), Some(node)) = (it.next(), it.next()) else {
            break;
        };
        if !(kind.starts_with('!') && node.starts_with('!')) {
            break;
        }
        let end = p + 2 + kind.len() + 1 + node.len();
        s.replace_range(p..end, "");
    }
    let words: Vec<&str> = s
        .split(' ')
        .filter(|w| {
            !(w.starts_with('#') && w.len() > 1 && w[1..].chars().all(|c| c.is_ascii_digit()))
        })
        .collect();
    words.join(" ")
}

/// Turn the wrapper function generated for `i` into a standalone LLVM IR module, in the
/// style of an LLVM codegen test: the operands that the wrapper receives from an `asm!`
/// block become function parameters, and the value it hands to the final `asm!` block is
/// returned. `asm` (rustc's assembly for the function) is appended as a comment.
pub fn standalone_ir(m: &LlvmModule, i: &Intrinsic, asm: Option<&FuncAsm>) -> Option<String> {
    let body = m.functions.get(&i.name)?;
    // Unnamed values (`%0`) must be numbered sequentially, which no longer holds once the
    // asm calls are removed, so give them names.
    let numeric = |id: &str| {
        (!id.is_empty() && id.chars().all(|c| c.is_ascii_digit())).then(|| format!("v{id}"))
    };
    let lines: Vec<String> = body
        .iter()
        .map(|l| {
            let l = map_locals(l, numeric);
            match l.split_once(':') {
                Some((label, "")) if label.chars().all(|c| c.is_ascii_digit()) => {
                    format!("v{label}:")
                }
                _ => l,
            }
        })
        .collect();

    // Inputs: `%v0 = tail call <ty> asm sideeffect "# in: ..."` (+ `extractvalue`s for several).
    let mut params: Vec<(String, String)> = vec![]; // (type, old value name)
    let mut drop = HashSet::new();
    if let Some((idx, l)) = lines
        .iter()
        .enumerate()
        .find(|(_, l)| l.contains("asm sideeffect \"# in:"))
    {
        drop.insert(idx);
        let (lhs, rhs) = l.split_once(" = ")?;
        let res = lhs.trim().trim_start_matches('%').to_string();
        let ty = split_ir_type(
            rhs.trim()
                .strip_prefix("tail call ")
                .or(rhs.trim().strip_prefix("call "))?,
        )
        .0;
        if ty.starts_with('{') {
            let fields: Vec<&str> = ty[1..ty.len() - 1].split(',').map(str::trim).collect();
            let mut by_index = vec![None; fields.len()];
            let suffix = format!(" %{res}, ");
            for (j, l) in lines.iter().enumerate() {
                let Some((lhs, rhs)) = l.split_once(" = extractvalue ") else {
                    continue;
                };
                let Some(p) = rhs.find(&suffix) else { continue };
                let k: usize = rhs[p + suffix.len()..].trim().parse().ok()?;
                by_index[k] = Some(lhs.trim().trim_start_matches('%').to_string());
                drop.insert(j);
            }
            for (k, f) in fields.iter().enumerate() {
                params.push((
                    f.to_string(),
                    by_index[k].clone().unwrap_or_else(|| format!("unused{k}")),
                ));
            }
        } else {
            params.push((ty.to_string(), res));
        }
    }
    let rename: HashMap<String, String> = params
        .iter()
        .zip(&i.params)
        .map(|((_, old), p)| (old.clone(), p.name.to_string()))
        .collect();

    let mut ret_ty = "void".to_string();
    let mut out_body = vec![];
    let mut skip_ret_void = false;
    for (j, l) in lines.iter().enumerate() {
        if drop.contains(&j) {
            continue;
        }
        if skip_ret_void && l.trim() == "ret void" {
            skip_ret_void = false;
            continue;
        }
        skip_ret_void = false;
        if l.contains("asm sideeffect \"# out:") {
            // `tail call void asm sideeffect "# out: ...", "f"(<2 x i64> %r) ...`
            let args = &l[l.rfind("\"(")? + 2..];
            let args = &args[..args.rfind(')')?];
            let (ty, val) = split_ir_type(args);
            ret_ty = ty.to_string();
            out_body.push(format!("  ret {ty} {val}"));
            skip_ret_void = true;
            continue;
        }
        if l.trim_start().starts_with("; Function Attrs") {
            continue;
        }
        out_body.push(strip_ir_annotations(l));
    }
    let out_body: Vec<String> = out_body
        .iter()
        .map(|l| map_locals(l, |id| rename.get(id).cloned()))
        .collect();

    // Declarations of everything called.
    let mut decls = vec![];
    for l in &out_body {
        let mut rest = l.as_str();
        while let Some(p) = rest.find('@') {
            rest = &rest[p + 1..];
            let end = rest.find('(').unwrap_or(rest.len());
            if let Some(d) = m.declares.get(&rest[..end]) {
                let d = strip_ir_annotations(d);
                if !decls.contains(&d) {
                    decls.push(d);
                }
            }
        }
    }

    // Only the target attributes of the function matter; keep them, de-duplicated.
    let attr_id = body_define_attr(m, i)?;
    let attrs = m.attributes.get(&attr_id)?;
    let mut kept = vec!["nounwind".to_string()];
    for key in ["\"target-cpu\"=", "\"target-features\"="] {
        if let Some(p) = attrs.find(key) {
            let v = &attrs[p + key.len()..];
            let v = &v[1..v[1..].find('"')? + 1];
            let mut seen = vec![];
            for f in v.split(',') {
                if !seen.contains(&f) {
                    seen.push(f);
                }
            }
            kept.push(format!("{key}\"{}\"", seen.join(",")));
        }
    }

    let mut s = String::new();
    writeln!(s, "; {}", i.signature()).unwrap();
    let e = i.expect();
    let want = if e.named.is_empty() {
        "no instruction".to_string()
    } else {
        e.named.join("|")
    };
    writeln!(s, "; Rust test: {} => {want}", i.call("")).unwrap();
    writeln!(s, ";").unwrap();
    writeln!(
        s,
        "; Generated by crates/loongarch-asm-test from rustc's optimized LLVM IR."
    )
    .unwrap();
    writeln!(s, "; RUN: llc -O3 < %s").unwrap();
    if let Some(f) = asm {
        writeln!(s, ";\n; rustc output:").unwrap();
        for l in &f.lines {
            if !l.starts_with('#') {
                writeln!(s, ";   {l}").unwrap();
            }
        }
    }
    writeln!(s, "\ntarget datalayout = {}", m.datalayout).unwrap();
    writeln!(s, "target triple = {}\n", m.triple).unwrap();
    let ps: Vec<String> = params
        .iter()
        .map(|(ty, old)| format!("{ty} %{}", rename.get(old).unwrap_or(old)))
        .collect();
    writeln!(s, "define {ret_ty} @{}({}) #0 {{", i.name, ps.join(", ")).unwrap();
    for l in &out_body {
        writeln!(s, "{l}").unwrap();
    }
    writeln!(s, "}}").unwrap();
    if !decls.is_empty() {
        writeln!(s).unwrap();
        for d in &decls {
            writeln!(s, "{d}").unwrap();
        }
    }
    writeln!(s, "\nattributes #0 = {{ {} }}", kept.join(" ")).unwrap();
    Some(s)
}

/// The attribute group of the wrapper function `i` in `m`.
fn body_define_attr(m: &LlvmModule, i: &Intrinsic) -> Option<String> {
    m.define_attrs.get(&i.name).cloned()
}
