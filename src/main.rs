#![feature(advanced_slice_patterns, slice_patterns, concat_idents)]

extern crate libc;
extern crate rand;

mod function; 
mod lstm; 


#[allow(unused_imports)]
use function::{sq, dot, abs, sigmoid, Function, Constant};
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::ops::Deref;
use lstm::lstm;

//extern { fn init_cublas(); }

//static POOL: Vec<Matrix> = vec![];

fn main() {
    //unsafe { init_cublas() };
    let args = HashMap::new();

    /////DEMO 1
    //let x = Function::param("x", Constant::Scalar(1.));
    //let f = &x;
    //f.minimize(&args, 0.1, 1000);

    /////DEMO 2
    //let x = Function::param("x", Constant::Scalar(1.));
    //let f = sq(&x);
    //f.minimize(&args, 0.01, 1000);

    /////DEMO 3
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(3.));
    //let f = sq(&(x + a));
    //f.minimize(&args, 0.1, 1000);

    /////DEMO 4
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(2.));
    //let f = sq(&(sq(&x) - a));
    //f.minimize(&args, 0.1, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(2.));
    //let b = Function::constant(Constant::Scalar(1.));
    //let c = Function::constant(Constant::Scalar(10.));
    //let f = sq(&((&x - &a) * (&x + &b) - c));
    //f.minimize(&args, 0.01, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::matrix(2, 2, vec![

                                                  //5., -6.,
                                                  //2.,  8.

                                                  //]));
    //let a = Function::constant(Constant::Scalar(2.));
    //let b = Function::constant(Constant::Scalar(1.));
    //let c = Function::constant(Constant::Scalar(10.));
    //let f = sq(&((&x - &a) * (&x + &b) - c));
    //f.minimize(&args, 0.0001, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::matrix(2, 2, vec![

                                                  //0.01, 0.01,
                                                  //0.01, 0.01

                                                  //]));

    //let a = Function::constant(Constant::matrix(2, 2, vec![

                                                  //4., 3.,
                                                  //3., 2.

                                                  //]));
    //let b = Function::constant(Constant::matrix(2, 2, vec![

                                                //1., 0.,
                                                //0., 1.

                                                //]));

    //let f = sq(&(dot(&a, &x) - b));
    //f.minimize(&args, 0.01, 10000);

    let dim = 2;
    let dims = vec![dim, dim];
    let inputs = vec![
        Constant::random(dims.clone(), -0.1, 0.1),
        Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
    ];
    //let target = Function::constant(Constant::random(dims.clone(), -10., 10.));
    //println!("target: {}", &target);
    //let x = Function::constant(Constant::random(vec![dim, dim], -1.1, 1.1));
    let f = lstm(inputs);
    f.minimize(&args, 0.01, 10000);
}
