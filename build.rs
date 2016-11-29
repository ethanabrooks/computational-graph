use std::process::Command;
use std::env;
use std::path::Path;
use std::io::Write;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    //writeln!(&mut std::io::stderr(), "pwd = {:?}", Command::new("pwd").output().unwrap());

    assert!(Command::new("nvcc")
        .args(&["src/matrix.cu", "-c", "-Xcompiler", "-fPIC", "-lcublas", "-o"]) 
        .arg(&format!("{}/matrix.o", out_dir))
        .status().unwrap().success(), "nvcc src/matrix.cu");

    assert!(Command::new("nvcc")
        .args(&["src/op.cu", "-c", "-Xcompiler", "-fPIC", "-lcublas", "-o"]) 
        .arg(&format!("{}/op.o", out_dir))
        .status().unwrap().success(), "nvcc src/op.cu");

    assert!(Command::new("rm")
        .args(&["-f", "libmatrix.a"]) 
        .current_dir(&Path::new(&out_dir)) 
        .status().unwrap().success(), "rm");

    assert!(Command::new("ar")
        .args(&["crus", "libmatrix.a", "matrix.o", "op.o"]) 
        .current_dir(&Path::new(&out_dir)) 
        .status().unwrap().success(), "ar");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda-7.5/lib64");
    println!("cargo:rustc-link-lib=static=matrix");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
}
