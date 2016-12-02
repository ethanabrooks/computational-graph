extern crate libc;
extern crate num;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix, eval, 
               grad, assign_outputs, minimize, maximize};
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = Scalar(2.);
    println!("a: {}", &a);
    println!("a: {}", &a.abs());
    //println!("a: {}", &a.abs());
    //let x = param("x", Scalar(1.)); 
    //let mut args = HashMap::new();
    //let f = &x * &(&a + &x);
    //println!("f: {}", f);

    //args.insert("x", Scalar(3.));

    //assign_outputs(&f, &args);
    //println!("args: {:#?}", args);

    //if let Some(c) = eval(&f, &args) {
        //println!("eval: {}", c);
    //}

    //println!("grad x: {}", grad(&f, "x"));

    //println!("x: {}", &x);
    //minimize(&f, 1., 1);
    //println!("x: {}", &x);
}
