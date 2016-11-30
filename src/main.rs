extern crate libc;

mod function; 
mod constant; 

use function::{variable, scalar, matrix, eval, grad, assign_outputs};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = scalar(2.);
    let b = matrix(2, 2, vec![
                   1., 1.,
                   1., 1.]);
    println!("b: {}", b);
    let x = variable("x", vec![]);
    let f1 = &x + &b;
    let f2 = &x + &a;
    let f = &f1 + &f2;
    println!("f: {}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));
    args.insert("y", Scalar(5.));
    assign_outputs(&f, &args);
    println!("args: {:#?}", args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }
    println!("grad x: {}", grad(&f, "x"));
}

