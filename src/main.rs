extern crate libc;
extern crate lifeguard;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, param, scalar, matrix, dot, new_constant, abs, sigmoid};
#[allow(unused_imports)]
use constant::{Matrix, Constant};
#[allow(unused_imports)]
use constant::Constant::Scalar;
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::ops::Deref;

extern { fn init_cublas(); }

fn main() {
    unsafe { init_cublas() };
    //let a = scalar(-2.); 
    //let b = matrix(2, 2, vec![
                   //1., 0., 
                   //0., 1.]); 
    //let b = new_constant(Constant::new(vec![300, 2000], 4.));

    //unsafe { print_matrix(a) };

    //unsafe { print_matrix(y) };

    //println!("a * x: {}", &a * &x);
    //println!("x: {}", x);
    //let f = (&x + &b).abs() * (&z + &a);

    //let f = (dot(&x, &y) + b).abs();
    //let f = dot(&x, &b, false, false);
    //println!("f: {}", &f);

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
    //let y = input("y", vec![]);
    //println!("a * b: {}", &a * &b);
    //let a = scalar(2.); 
    //let b = scalar(3.); 
    let a = matrix(2, 2, vec![
                   1., 2., 
                  -3., 4.]); 

    let x = param("x", Constant::Matrix(Matrix::new(2, 2, vec![
                   1., 3., 
                   1., 1.]))); 

    let b = matrix(2, 2, vec![
                   5., 3.,
                   5., 1.]); 

    //let y = param( "y", Constant::Matrix(Matrix::new(3, 2, vec![
                                    //1., 1., 
                                    //7., 1.,
                                    //2., 5.]))); 


    //let x = param("x", Scalar(0.9));
    //let y = param("y", Scalar(0.9));

    //let f = abs(&(dot(&a, &x, false, false) - b));
    let f = abs(&(dot(&a, &x) - b));
    let mut args = HashMap::new();
    args.insert("y", Scalar(-3.));

    f.minimize(&args, 0.1, 150);

    println!("f: {}", &f);
    println!("eval f: {}", &f.eval(&args));
    //println!("output f: {}", &f.unwrap_value().deref());

    //println!("{}", Constant::Matrix(Matrix::new(2, 3, vec![
                   //1., 2., 3., 
                   //1., 2., 1.]))); 
    //println!("output: {}", f.eval(&args));
}
