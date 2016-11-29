extern crate libc;

mod function; 
mod constant; 

use function::{variable, scalar, matrix, eval, grad, assign_outputs};
use constant::Constant::Scalar;
use constant::Matrix;
use std::collections::HashMap;
use std::ptr;

extern "C" {
    fn double_input(input: libc::c_int) -> libc::c_int;
}

//fn main() {
    //let input = 4;
    //let output = unsafe { double_input(input) };
    //println!("{} * 2 = {}", input, output);
//}

fn main() {
    let a = scalar(2.);
    let b = matrix(2, 2, vec![
                   1., 2.,
                   3., 4.]);
    println!("b: {}", b);
    let x = variable("x", vec![]);
    let f = &x + &b;
    println!("f: {}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));
    args.insert("y", Scalar(5.));
    assign_outputs(&f, &args);
    println!("args: {:#?}", args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }
    //println!("grad x: {}", grad(&f, "x"));
}

