extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix, eval, 
               grad, assign_outputs, minimize, maximize};
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

    assign_outputs(&f, &args);
    println!("args: {:#?}", args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }

    println!("grad x: {}", grad(&f, "x"));

    //println!("x: {}", &x);
    minimize(&f, 1., 1);
    //println!("x: {}", &x);
}
