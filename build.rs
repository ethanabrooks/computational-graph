use std::process::Command;
use std::env;
use std::path::Path;


fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("nvcc").args(&["src/main.cpp src/matrix.cu", "-c", "-fPIC", "-o"]) 
        .arg(&format!("{}/main.o", out_dir)) 
        .status().unwrap(); 

    Command::new("ar").args(&["crus", "libmatrix.a", "main.o"]) 
        .current_dir(&Path::new(&out_dir)) 
        .status().unwrap();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=matrix");
}
