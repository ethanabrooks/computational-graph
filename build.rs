use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let names = vec!["matrix", "op", "scan"];

    let src_files: Vec<String> = names.clone().into_iter()
        .map(|name| format!("src/{}.cu", name))
        .collect();

    let out_files: Vec<String> = names.clone().into_iter()
        .map(|name| format!("{}/{}.o", out_dir, name))
        .collect();

    for i in 0..names.len() {
        assert!(Command::new("nvcc")
            .args(&[src_files[i].as_str(), "-c", "-Xcompiler", "-fPIC", "-lcublas", "-o"]) 
            .arg(&out_files[i])
            .status().unwrap().success(), "nvcc src/matrix.cu");
    }

    assert!(Command::new("rm")
        .args(&["-f", "libmatrix.a"]) 
        .current_dir(&Path::new(&out_dir)) 
        .status().unwrap().success(), "rm");

    assert!(Command::new("ar")
        .args(&["crus", "libmatrix.a", "matrix.o", "op.o", "scan.o"]) 
        .current_dir(&Path::new(&out_dir)) 
        .status().unwrap().success(), "ar");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda-7.5/lib64");
    println!("cargo:rustc-link-lib=static=matrix");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
