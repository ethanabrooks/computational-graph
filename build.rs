use std::process::Command;
use std::env;
use std::path::Path;
use std::fs;
use std::time::SystemTime;
use std::io::Write;

fn more_recent_than(srcs: Vec<String>, dst: &str) -> std::io::Result<bool> {
    match fs::metadata(dst) {
        Ok(metadata) => {
            let time_mod_dst = metadata.modified()?;
            for src in srcs {
                let time_mod_src = fs::metadata(src)?.modified()?;

                if time_mod_src > time_mod_dst {
                    return Ok(true)
                }
            } 
            return Ok(false)
        }
        _ => Ok(true)
    }
}

fn main() {
    let out_dir: String = env::var("OUT_DIR").unwrap();

    let out_name = |name| format!("{}/{}.o", out_dir, name);

    let c_names = vec!["matrix", "op", "scan"];
    for i in 0..c_names.len() {
        let src_name = format!("src/{}.cu", c_names[i]);
        if more_recent_than(vec![src_name.clone()], &out_name(c_names[i])).unwrap() {
            assert!(Command::new("nvcc")
                .arg(&src_name)
                .args(&["-c", "-Xcompiler", "-fPIC", "-lcublas", "-o"]) 
                .arg(&out_name(c_names[i]))
                .status().unwrap().success(), "nvcc src/matrix.cu");
        }
    }

    let out_files: Vec<String> = c_names.into_iter().map(out_name).collect();

    if more_recent_than(out_files, "libmatrix.a").unwrap() {
        assert!(Command::new("rm")
            .args(&["-f", "libmatrix.a"]) 
            .current_dir(&Path::new(&out_dir)) 
            .status().unwrap().success(), "rm");

        assert!(Command::new("ar")
            .args(&["crus", "libmatrix.a", "matrix.o", "op.o", "scan.o"]) 
            .current_dir(&Path::new(&out_dir)) 
            .status().unwrap().success(), "ar");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda-7.5/lib64");
    println!("cargo:rustc-link-lib=static=matrix");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
