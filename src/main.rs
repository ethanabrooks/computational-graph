extern crate libc;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{input, param, scalar, matrix, dot, new_constant};
#[allow(unused_imports)]
use constant::{Matrix, Constant};
#[allow(unused_imports)]
use constant::Constant::Scalar;
#[allow(unused_imports)]
use std::collections::HashMap;

extern { fn init_cublas(); }

fn main() {
    unsafe { init_cublas() };
    //let a = scalar(-2.); 
    //let b = scalar(-3.); 
    //let y = input("y", vec![]);
    //println!("a * b: {}", &a * &b);
    //let x = param("x", Scalar(14.));
    //let z = param("z", Scalar(10.));
    let x = param("x", Constant::new(vec![2000, 300], 1.));
                   //1., 2., 3., 
                   //1., 2., 1.])); 
    //let b = matrix(2, 2, vec![
                   //1., 0., 
                   //0., 1.]); 
    let b = new_constant(Constant::new(vec![300, 2000], 4.));

    //let x = param( "x", new_matrix( 2, 3, vec![
                                    //1., 1., 1., 
                                    //1., 1., 1.])); 

    //unsafe { print_matrix(a) };

    //let y = matrix(2, 2, vec![
                   //2., 2., 
                   //2., 2.]); 
    let y = matrix(3, 2, vec![
                   2., 2., 
                   2., -2.,
                   2., 2.]); 

    //unsafe { print_matrix(y) };

    //println!("a * x: {}", &a * &x);
    //println!("x: {}", x);
    //let f = (&x + &b).abs() * (&z + &a);

    //let f = (dot(&x, &y) + b).abs();
    let f = dot(&x, &b);
    //let f = x .sigmoid();
    //println!("f: {}", &f);

    let mut args = HashMap::new();
    args.insert("y", Scalar(-3.));
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

    f.minimize(&args, 0.1, 1000);

    //println!("f: {}", &f);
    //println!("output: {}", f.eval(&args));
}
