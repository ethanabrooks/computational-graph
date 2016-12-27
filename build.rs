use std::process::Command;
use std::{env, fs, str};
use std::path::Path;


fn more_recent_than(srcs: &Vec<String>, dst: &str) -> std::io::Result<bool> {
    match fs::metadata(dst) {
        Ok(metadata_dst) => {
            let time_mod_dst = metadata_dst.modified()?;
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
    let c_ext = match Command::new("nvcc").status() {
        Ok(_) => "cu",
        Err(_) => "cpp",
    };
    let c_names = vec!["matrix", "ops", "util"];
    let dir = "src/c";

    let out_dir = env::var("OUT_DIR").expect("WTF 1");
    let get_out_name = |name| format!("{}/{}.o", out_dir, name);

    for i in 0..c_names.len() {
        let src_name = format!("{}/{}.{}", dir, c_names[i], c_ext);
        let out_name = get_out_name(c_names[i]);

        if more_recent_than(&vec![src_name.clone()], &out_name).expect("WTF 2") {
            assert!(Command::new("nvcc")
                .arg(&src_name)
                .args(&["-c", "-Xcompiler", "-fPIC", "-lcublas", "-o"]) 
                .arg(&out_name)
                .status().expect("WTF 3").success(), "nvcc {} failed", src_name);
        }
    }

    let out_files: Vec<String> = c_names.into_iter().map(get_out_name).collect();

    if more_recent_than(&out_files, "libmatrix.a").expect("WTF 4") {

        assert!(Command::new("rm")
            .args(&["-f", "libmatrix.a"]) 
            .current_dir(&Path::new(&out_dir)) 
            .status().expect("WTF 5").success(), "rm failed");


        assert!(Command::new("ar")
            .args(&["crus", "libmatrix.a"])
            .args(&out_files)
            .current_dir(&Path::new(&out_dir)) 
            .status().expect("WTF 6").success(), "ar failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    if let Some(paths) = env::var_os("LD_LIBRARY_PATH") {
        for path in env::split_paths(&paths) {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    println!("cargo:rustc-link-lib=static=matrix");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
