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

extern { fn init_cublas(); }

//static POOL: Vec<Matrix> = vec![];

fn main() {
    unsafe { init_cublas() };
    let args = HashMap::new();
    let dim = 20;
    //let dims = vec![20, 20];
    //let inputs = vec![
        //Constant::random(dims.clone(), -0.1, 0.1),
        //Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
    //];
    //let target = Function::constant(Constant::random(dims.clone(), -10., 10.));
    //println!("target: {}", &target);
    let x = Function::constant(Constant::random(vec![dim, dim], -0.1, 0.1));
    let f = dot(&x, &x);
    f.minimize(&args, 0.01, 1000);

    //println!("f: {}", &f);
}
