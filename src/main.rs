extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix};
use constant::Constant::Scalar;
use constant::new_matrix;
use std::collections::HashMap;

fn main() {
    let a = scalar(-2.); 
    let b = scalar(-3.); 
    let y = input("y", vec![]);
    //println!("a * b: {}", &a * &b);
    let x = param( "x", new_matrix(
            2, 3, vec![
            12., 12., 12., 
            12., 12., 12.])); 
    //println!("a * x: {}", &a * &x);
    println!("x: {}", x);
    //let f = (&x + &b).abs() * (&x + &a);
    let f = a + y;

    let mut args = HashMap::new();
    args.insert("y", Scalar(-3.));
    let g = f.clone();
    f.assign_outputs(&args);
    println!("f.output: {}", f.get_output());
    println!("g.output: {}", g.get_output());

    args.insert("y", Scalar(10.));
    f.assign_outputs(&args);
    println!("f.output: {}", f.get_output());
    println!("g.output: {}", g.get_output());
    //println!("args: {:#?}", args);

    //if let Some(c) = f.eval(&args) {
        //println!("eval: {}", c);
    //}

    //println!("grad x: {}", f.grad("x"));

    //println!("x: {}", &x);
    //f.minimize(&args, 0.0005, 20);
    //println!("x: {}", &x);
}
