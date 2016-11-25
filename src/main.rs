#![feature(box_syntax, box_patterns)]

mod function; 
mod constant; 
use function::{variable, scalar, eval, grad, assign_outputs};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = &scalar(2.);
    //let b = &scalar(3.);
    let x = &variable("x", vec![]);
    //let y = &variable("y", vec![]);

    let f1 = -x;
    let f2 = &f1 * a;
    let f = &f2 * a;
    println!("f: {}", f);
    //let mut g = -x;

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));
    args.insert("y", Scalar(5.));
    assign_outputs(&f, &args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }
    println!("grad x: {}", grad(&f, "x"));
}

