#![feature(advanced_slice_patterns, slice_patterns, concat_idents, plugin)]
#![cfg_attr(test, plugin(stainless))]

#[macro_use]
extern crate lazy_static;
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
#[allow(unused_imports)]
use lstm::lstm;

// TODO: design wrapper for matrix ops that checks for this.
extern { fn init_cublas(); }

pub fn init() {
    unsafe { init_cublas() };
}

fn main() {
    unsafe { init_cublas() };
    let args = HashMap::new();

    ///DEMO 1
    let x = Function::param("x", Constant::Scalar(1.));
    let f = &x;
    f.minimize(&args, 1., 10);

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

    //let dim = 5;
    //let dims = vec![dim, dim];
    //let inputs = vec![
        //Constant::random(dims.clone(), -0.1, 0.1),
        //Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
        ////Constant::random(dims.clone(), -0.1, 0.1),
    //];
    //let target = Function::constant(Constant::random(dims.clone(), -10., 10.));
    //println!("target: {}", &target);
    ////let x = Function::constant(Constant::random(vec![dim, dim], -1.1, 1.1));
    //let f = sq(&(lstm(inputs) - target));
    //f.minimize(&args, 0.01, 100000);

    println!("f: {}", f);
}


describe! tests {
    before_each {
        macro_rules! make_test {
        }
    }

    make_test!();

    after_each {
    }
}

/*
describe! tests {
    before_each {
        init();
        let args = HashMap::new();
        let fx;
        let fX;
        macro_rules! make_test {
            ($f:expr) => {
                it std::fmt("minimizes {}") {
                    fx = $f(Function::param("x", Constant::Scalar(1.)));
                    fX = $f(Function::param("X", Constant::single_val(vec![2, 2], 1.)));
                }
            }
        }
    }

    it "minimizes single parameters" {
        make_functions!(|x| x);
    }

    //it "minimizes negatives" {
        //make_functions!(|x: Function| -x);
    //}

    //it "minimizes squares" {
        //make_functions!(|x: Function| sq(x));
    //}

    after_each {
        for f in &[fx, fX] {
            f.minimize(&args, 1., 10);
            assert!(f.all_less_than(0.01));
        }
    }
}
*/
