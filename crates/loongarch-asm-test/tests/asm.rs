//! Compiles a wrapper for every LSX/LASX intrinsic for `loongarch64-unknown-linux-gnu`
//! (cross-compiling on other hosts; nothing is executed), prints the generated assembly
//! of each one and checks that it is just the instruction the intrinsic is named after
//! (or an equivalent single instruction). The test fails if any intrinsic is lowered to
//! something else, or needs additional instructions.
//!
//! ```text
//! cargo test -r -p loongarch-asm-test                     # all intrinsics
//! cargo test -r -p loongarch-asm-test -- vadd xvpermi     # only names containing a filter
//! cargo test -r -p loongarch-asm-test -- --list           # just list intrinsics + signatures
//! cargo test -r -p loongarch-asm-test -- --quiet          # only the summary
//! cargo test -r -p loongarch-asm-test -- --llvm-ir        # print LLVM IR instead of asm and
//!                                                         # write one llc-ready .ll per intrinsic
//! cargo test -r -p loongarch-asm-test -- --llvm-ir=DIR    # same, writing the .ll files to DIR
//! ```

use loongarch_asm_test::*;
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const TARGET: &str = "loongarch64-unknown-linux-gnu";

struct Opts {
    list: bool,
    quiet: bool,
    /// Directory for per-intrinsic `.ll` files, when `--llvm-ir` is given.
    llvm_ir: Option<PathBuf>,
    filters: Vec<String>,
}

fn parse_args() -> Opts {
    let mut o = Opts {
        list: false,
        quiet: false,
        llvm_ir: None,
        filters: vec![],
    };
    for a in env::args().skip(1) {
        match a.as_str() {
            "--list" => o.list = true,
            "--quiet" | "-q" => o.quiet = true,
            "--llvm-ir" => o.llvm_ir = Some(work_dir().join("llvm-ir")),
            s if s.starts_with("--llvm-ir=") => {
                o.llvm_ir = Some(PathBuf::from(&s["--llvm-ir=".len()..]))
            }
            // Ignore libtest flags such as `--nocapture` that may be passed along.
            s if s.starts_with('-') => {}
            s => o.filters.push(s.to_string()),
        }
    }
    o
}

/// Where the wrapper crate and its outputs live.
fn work_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("loongarch-asm-test")
}

fn write_if_changed(path: &Path, contents: &str) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(contents) {
        std::fs::write(path, contents)
            .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    }
}

fn target_installed() -> bool {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    Command::new(rustc)
        .args(["--print", "target-libdir", "--target", TARGET])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()).is_dir())
        .unwrap_or(false)
}

/// Generate the wrapper crate and compile it; returns the assembly and LLVM IR text.
fn build(repo_root: &Path, intrinsics: &[Intrinsic]) -> Result<(String, String), String> {
    let dir = work_dir();
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let core_arch = repo_root.join("crates/core_arch");
    write_if_changed(
        &dir.join("Cargo.toml"),
        &format!(
            r#"[package]
name = "loongarch-asm-wrappers"
version = "0.0.0"
edition = "2024"

[lib]
path = "src/lib.rs"

[dependencies]
core_arch = {{ path = {core_arch:?} }}

[profile.release]
opt-level = 3
debug = false
codegen-units = 1
incremental = false
panic = "abort"

[workspace]
"#
        ),
    );
    // Cargo only re-runs rustc (and so only re-emits the outputs) when the crate is stale.
    let asm_path = dir.join("wrappers.s");
    let ir_path = dir.join("wrappers.ll");
    let src = dir.join("src/lib.rs");
    let wrappers = gen_wrappers(intrinsics);
    if asm_path.exists() && ir_path.exists() {
        write_if_changed(&src, &wrappers);
    } else {
        std::fs::write(&src, &wrappers).unwrap();
    }
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(&dir)
        // Keep the build independent of whatever flags the outer build uses.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .args([
            "rustc",
            "--quiet",
            "--release",
            "--target",
            TARGET,
            "--target-dir",
        ])
        .arg(dir.join("target"))
        .args(["--", "-Zmerge-functions=disabled", "--emit"])
        .arg(format!(
            "asm={},llvm-ir={}",
            asm_path.display(),
            ir_path.display()
        ))
        .status()
        .map_err(|e| format!("failed to run cargo: {e}"))?;
    if !status.success() {
        return Err(format!("compiling {} failed", dir.display()));
    }
    let read =
        |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()));
    Ok((read(&asm_path)?, read(&ir_path)?))
}

fn main() -> ExitCode {
    let opts = parse_args();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let intrinsics = load_intrinsics(&repo_root);
    let selected: Vec<&Intrinsic> = intrinsics
        .iter()
        .filter(|i| {
            opts.filters.is_empty() || opts.filters.iter().any(|f| i.name.contains(f.as_str()))
        })
        .collect();

    if opts.list {
        for i in &selected {
            let imp = if i.portable { "portable" } else { "llvm" };
            println!("{:<4} {:<8} {}", i.ext.name(), imp, i.signature());
        }
        println!("\n{} intrinsics", selected.len());
        return ExitCode::SUCCESS;
    }

    if !target_installed() {
        println!(
            "SKIPPED: the `{TARGET}` standard library is not installed.\n\
             Install it with `rustup target add {TARGET}` to run this test."
        );
        return ExitCode::SUCCESS;
    }

    let (asm, ir) = match build(&repo_root, &intrinsics) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let names: HashSet<&str> = intrinsics.iter().map(|i| i.name.as_str()).collect();
    let funcs = parse_asm(&asm, &names);
    let module = opts.llvm_ir.as_ref().map(|dir| {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("creating {}: {e}", dir.display()));
        parse_llvm_ir(&ir)
    });
    let mut ir_files = 0;

    let mut counts = [0usize; 4];
    let mut bad: Vec<(&str, &str)> = vec![];
    for i in &selected {
        let e = i.expect();
        let Some(f) = funcs.get(&i.name) else {
            println!("{}: <no assembly found>\n", i.name);
            bad.push((i.name.as_str(), "no assembly"));
            continue;
        };
        let v = verdict(&e, f);
        counts[v as usize] += 1;
        let tag = match v {
            Verdict::Exact => "ok",
            Verdict::Equivalent => "ok, equivalent instruction",
            Verdict::Extra => "EXTRA INSTRUCTIONS",
            Verdict::Missing => "MISSING INSTRUCTION",
        };
        if matches!(v, Verdict::Extra | Verdict::Missing) {
            bad.push((i.name.as_str(), tag));
        }
        if let (Some(dir), Some(m)) = (&opts.llvm_ir, &module) {
            match standalone_ir(m, i, Some(f)) {
                Some(text) => {
                    write_if_changed(&dir.join(format!("{}.ll", i.name)), &text);
                    ir_files += 1;
                    if !opts.quiet {
                        println!("; [{tag}]\n{text}");
                    }
                }
                None => println!("{}: <could not extract LLVM IR>\n", i.name),
            }
            continue;
        }
        if !opts.quiet {
            let want = if e.named.is_empty() {
                "no instruction".to_string()
            } else {
                e.named.join("|")
            };
            println!("// {} => {} [{}]", i.call(""), want, tag);
            println!("{}:", i.name);
            for l in &f.lines {
                if l.ends_with(':') {
                    println!("{l}");
                } else {
                    println!("    {l}");
                }
            }
            println!();
        }
    }

    println!(
        "{} intrinsics: {} exact, {} equivalent, {} with extra instructions, {} missing the instruction",
        selected.len(),
        counts[Verdict::Exact as usize],
        counts[Verdict::Equivalent as usize],
        counts[Verdict::Extra as usize],
        counts[Verdict::Missing as usize],
    );
    if let Some(dir) = &opts.llvm_ir {
        println!("wrote {ir_files} LLVM IR files to {}", dir.display());
    }
    if bad.is_empty() {
        return ExitCode::SUCCESS;
    }
    println!("\nNot lowered to the ideal instruction:");
    for (name, why) in &bad {
        println!("    {name:<28} {why}");
    }
    ExitCode::FAILURE
}
