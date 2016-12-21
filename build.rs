use std::process::Command;
use std::env;
use std::path::Path;
use std::fs;

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
    let c_names = vec!["matrix", "ops", "util"];

    let out_dir = env::var("OUT_DIR").unwrap();
    let get_out_name = |name| format!("{}/{}.o", out_dir, name);

    for i in 0..c_names.len() {
        let src_name = format!("src/{}.cpp", c_names[i]);
        let out_name = get_out_name(c_names[i]);
         
        if more_recent_than(&vec![src_name.clone()], &out_name).unwrap() {
            assert!(Command::new("c++")
                .arg(&src_name)
                .args(&["-c", "-std=c++11", "-o"]) //"-Xcompiler", "-fPIC", "-std=c++11"]) 
                .arg(&out_name)
                .status().unwrap().success(), "c++ {} failed", src_name);
        }
    }

    let out_files: Vec<String> = c_names.into_iter().map(get_out_name).collect();

    if more_recent_than(&out_files, "libmatrix.a").unwrap() {

        assert!(Command::new("rm")
            .args(&["-f", "libmatrix.a"]) 
            .current_dir(&Path::new(&out_dir)) 
            .status().unwrap().success(), "rm failed");


        assert!(Command::new("ar")
            .args(&["crus", "libmatrix.a"])
            .args(&out_files)
            .current_dir(&Path::new(&out_dir)) 
            .status().unwrap().success(), "ar failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=matrix");
    //println!("cargo:rustc-link-search=native={}", "/usr/local/cuda-7.5/lib64");
    //println!("cargo:rustc-link-lib=dylib=cublas");
    //println!("cargo:rustc-link-lib=dylib=cudart");
}
