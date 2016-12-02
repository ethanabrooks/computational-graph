extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = scalar(-2.); 
    let b = scalar(-3.); 
    println!("a * b: {}", &a * &b);
    let x = param("x", Scalar(-3.)); 
    println!("a * x: {}", &a * &x);
    let f = x;
    println!("f: {}", f);

    let mut args = HashMap::new();
    //args.insert("x", Scalar(-3.));

    f.assign_outputs(&args);
    println!("args: {:#?}", args);

    if let Some(c) = f.eval(&args) {
        println!("eval: {}", c);
    }

    println!("grad x: {}", f.grad("x"));

    //println!("x: {}", &x);
    f.minimize(1., 10);
    //println!("x: {}", &x);
}
