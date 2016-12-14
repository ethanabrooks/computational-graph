#![feature(advanced_slice_patterns, slice_patterns)]

extern crate libc;
extern crate rand;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{sq, dot, abs, sigmoid, rnn, Function, lstm};
#[allow(unused_imports)]
use constant::{Matrix, Constant};
#[allow(unused_imports)]
use constant::Constant::Scalar;
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::ops::Deref;

//extern { fn init_cublas(); }

//static POOL: Vec<Matrix> = vec![];

fn main() {
    //unsafe { init_cublas() };
    let args = HashMap::new();

    /////DEMO 1
    //let x = Function::param("x", Constant::Scalar(1.));
    //let f = &x;
    //f.slow_minimize(&args, 0.1, 1000);

    /////DEMO 2
    //let x = Function::param("x", Constant::Scalar(1.));
    //let f = sq(&x);
    //f.minimize(&args, 0.01, 1000);

    /////DEMO 3
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(3.));
    //let f = sq(&(x + a)); // (x + 3)^2
    //f.minimize(&args, 0.1, 1000);

    /////DEMO 4
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(2.));
    //let f = sq(&(sq(&x) - a)); // (x^2 - 2)^2 
    //f.minimize(&args, 0.1, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::Scalar(1.));
    //let a = Function::constant(Constant::Scalar(2.));
    //let b = Function::constant(Constant::Scalar(1.));
    //let c = Function::constant(Constant::Scalar(10.));
    //let f = sq(&((&x - &a) * (&x + &b) - c));
    //// ((x-2)(x+1) - 10)^2
    //// (x-2)(x+1) = 10
    //f.minimize(&args, 0.01, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::random(vec![2, 2], -10., 10.));
    //let a = Function::constant(Constant::Scalar(2.));
    //let b = Function::constant(Constant::Scalar(1.));
    //let c = Function::constant(Constant::Scalar(10.));
    //let f = sq(&((&x - &a) * (&x + &b) - c));
    //f.minimize(&args, 0.0001, 1000);

    /////DEMO 5
    //let x = Function::param("x", Constant::random(vec![2, 2], -1., 1.));
    //let a = Function::constant(Constant::matrix(2, 2, vec![

                                                  //4., 3.,
                                                  //3., 2.

                                                  //]));
    //let b = Function::constant(Constant::matrix(2, 2, vec![

                                                //1., 0.,
                                                //0., 1.

                                                //]));

    //let f = sq(&(dot(&a, &x) - b));
    //// (a . x - I)^2
    //// a . x = I
    //f.minimize(&args, 0.001, 1000);


    let dims = vec![10, 10];
    let inputs = vec![
        Constant::random(dims.clone(), -0.1, 0.1),
        Constant::random(dims.clone(), -0.1, 0.1),
        Constant::random(dims.clone(), -0.1, 0.1),
        Constant::random(dims.clone(), -0.1, 0.1),
    ];
    let target = Function::constant(Constant::random(dims.clone(), -10., 10.));
    println!("target: {}", &target);
    let f = sq(&(lstm(inputs) - target));
    f.minimize(&args, 0.01, 100000);

    println!("f: {}", &f);
}
