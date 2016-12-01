extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix, eval, 
               grad, assign_outputs, minimize, maximize};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    //let a = scalar(2.);
    let a = matrix(1, 2, vec![1., 2.]);
    println!("a: {}", a);
    let b = matrix(2, 2, vec![
                   1., 2.,
                   3., 4.]);
    println!("b: {}", b);
    let c = matrix(3, 2, vec![
                   1., 2.,
                   3., 4.,
                   5., 6.]);
    println!("c: {}", c);
    let x = param("x", Scalar(1.));
    let f = (&a + &a) + a;
    println!("f: {}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));

    assign_outputs(&f, &args);
    println!("args: {:#?}", args);

    if let Some(c) = eval(&f, &args) {
        println!("eval: {}", c);
    }

    println!("grad x: {}", grad(&f, "x"));

    println!("x: {}", x);
    minimize(&f, 1., 1);
    println!("x: {}", x);
}
