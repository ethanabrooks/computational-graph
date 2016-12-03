extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, scalar, param, matrix};
use constant::Constant::Scalar;
use constant::{new_matrix};
use std::collections::HashMap;

fn main() {
    //let a = scalar(-2.); 
    //let b = scalar(-3.); 
    //let y = input("y", vec![]);
    //println!("a * b: {}", &a * &b);
    //let x = param( "x", Scalar(14.));
    let a = new_matrix(2, 3, vec![
                       1., 1., 1., 
                       1., 1., 1.]); 
    let b = new_matrix(3, 2, vec![
                       2., 2., 
                       2., 2.,
                       2., 2.]); 

    let x = param( "x", new_matrix( 2, 3, vec![
                                    1., 1., 1., 
                                    1., 1., 1.])); 
    let y = matrix(3, 2, vec![
                   2., 2., 
                   2., 2.,
                   2., 2.]); 

    //println!("a * x: {}", &a * &x);
    //println!("x: {}", x);
    //let f = (&x + &b).abs() * (&x + &a);
    //let f = x .sigmoid();
    //println!("f: {}", &f);

    //let mut args = HashMap::new();
    //args.insert("y", Scalar(-3.));
    //let g = f.clone();
    //f.assign_outputs(&args);
    //println!("f.output: {}", f.get_output());
    //println!("g.output: {}", g.get_output());

    //args.insert("y", Scalar(10.));
    //f.assign_outputs(&args);
    //println!("f.output: {}", f.get_output());
    //println!("g.output: {}", g.get_output());
    //println!("args: {:#?}", args);

    //if let Some(c) = f.eval(&args) {
        //println!("eval: {}", c);
    //}

    //println!("grad x: {}", f.grad("x"));

    //f.minimize(&args, 20., 100000);
    //println!("x: {}", &x);
}
