extern crate libc;

mod function; 
mod constant; 

use function::{input, scalar, param, matrix, eval, grad, assign_outputs, backprop};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = scalar(2.);
    let b = matrix(2, 2, vec![
                   1., 2.,
                   3., 4.]);
    println!("b: {}", b);
    let x = param("x", Scalar(1.));
    let f1 = &x + &b;
    let f2 = &x + &b;
    let f = &f1 * &x;
    println!("f: {}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));

    assign_outputs(&f, &args);
    println!("args: {:#?}", args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }

    backprop(&f, &f.output.borrow().clone().unwrap(), 1.);

    println!("grad x: {}", grad(&f, "x"));
}

