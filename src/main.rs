#![feature(advanced_slice_patterns, slice_patterns, concat_idents, plugin)]
#![cfg_attr(test, plugin(stainless))]

#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate rand;

mod function; 
mod lstm; 

#[allow(unused_imports)]
use function::{sq, dot, abs, sigmoid, tanh, Function, Constant};
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
    f.minimize(&args, 1., 10, 1);

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

#[cfg(test)]
mod tests {
    use super::*;
    const START: f32 = 0.5;

    macro_rules! test {
        ($test_name:ident, $f:expr, $learn_rate:expr, $goal:expr) => {
            #[test]
            fn $test_name () {
                init();
                let x = Function::param("x", Constant::Scalar(START));
                let m = Function::param("m", Constant::single_val(vec![2, 2], START));
                for f in &[$f(x), $f(m)] {
                    f.minimize(&HashMap::new(), $learn_rate, 10, 1);
                    assert!(f.all_less_than($goal), "F: {}", f);
                }
            }
        };
        ($test_name:ident, $f:expr, $const_val:expr, $learn_rate:expr, $goal:expr) => {
            #[test]
            fn $test_name () {
                init();
                let x = Function::param("x", Constant::Scalar(START));
                let m = Function::param("m", Constant::single_val(vec![2, 2], START));
                for f in &[$f(x.clone(), x.clone()),
                           $f(x.clone(), m.clone()),
                           $f(m.clone(), x.clone()),
                           $f(m.clone(), m.clone())] {
                    f.minimize(&HashMap::new(), $learn_rate, 10, 1);
                    assert!(f.all_less_than($goal), "F: {}", f);
                }
            }
        }
    }

    test!(single_param_test, |x| x, 1., -9.4);
    test!(neg_test, |x: Function| -x, 1., -9.4);
    test!(sq_test, |x: Function| sq(x), 0.1, 0.1);
    test!(abs_test, |x: Function| abs(x), 0.1, 0.1);
    test!(sigmoid_test, |x: Function| sigmoid(x), 1., 0.2);
    test!(tanh_test, |x: Function| tanh(x), 1., -0.9);
    test!(add_test, |x: Function, y: Function| x + y, 1., 1., -34.);
    test!(mul_test, |x: Function, y: Function| x * y, 2., 0.1, 0.1);
    test!(complex_test, |x: Function| {
        let a = Function::scalar(100.);
        let b = Function::matrix(2, 2, vec![
                                8., -5.,
                                0., 10.
                                ]);
        let c = Function::scalar(10.);
        sq(((&x - &a) * (&x + &b) - c))
    }, 0.0001, 0.00001);
    test!(sub_test, |x: Function, y: Function| x - y, 2., 0.01, -34.);
    //test!(dot_test, |x: Function, y: Function| dot(&x, &y), 1., 1., -34.);
}
