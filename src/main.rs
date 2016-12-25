#![feature(advanced_slice_patterns, slice_patterns, concat_idents, test, drop_types_in_const)]
#![cfg_attr(test, plugin(stainless))]

#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate test;

mod function; 
mod constant; 
mod matrix; 
mod ops; 
mod optimize; 
mod print; 
mod lstm; 

#[allow(unused_imports)]
use ops::{init, sq, dot, abs, sigmoid, tanh};
#[allow(unused_imports)]
use function::Function;
#[allow(unused_imports)]
use constant::Constant;
#[allow(unused_imports)]
use lstm::lstm;
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::ops::Deref;
#[allow(unused_imports)]
use matrix::PMatrix;

// TODO: design wrapper for matrix ops that checks for this.
fn main() {
    init();
    let dim = 10;
    let dims = vec![dim, dim];
    //let inputs = vec![
        //Constant::random(dims.clone(), -0.1, 0.1),
        //Constant::random(dims.clone(), -0.1, 0.1)
        //]; 
    //let target = Function::random_matrix(dims.clone(), -0.1, 0.1);
    //let f = sq(lstm(inputs) - target);
    //let m = Function::single_val_matrix(2, 2, 0.1);
    let f = Function::random_param("x", dims.clone(), -0.1, 0.1);
    f.minimize(&HashMap::new(), 0.01, 100, 1);
    //PMatrix::empty(2, 2);
    //println!("HERE");
    //PMatrix::empty(2, 2);
    //println!("HERE");
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

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
                let cx = Function::scalar($const_val);
                let cm = Function::single_val_matrix(2, 2, $const_val);
                let px = Function::param("x", Constant::Scalar(START));
                let pm = Function::param("m", Constant::single_val(vec![2, 2], START));
                for f in &[$f(cx.clone(), px.clone()),
                           $f(cx.clone(), pm.clone()),
                           $f(cm.clone(), px.clone()),
                           $f(cm.clone(), pm.clone()),
                           $f(px.clone(), cx.clone()),
                           $f(px.clone(), cm.clone()),
                           $f(pm.clone(), cx.clone()),
                           $f(pm.clone(), cm.clone())] {
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
    test!(add_test, |x: Function, y: Function| x + y, 1., 1., -7.4);
    test!(mul_test, |x: Function, y: Function| x * y, 2., 0.1, -2.5);
    test!(sub_scalar_l_test, |x: Function| Function::scalar(1.) - x, 1., -8.4);
    test!(sub_matrix_r_test,
          |x: Function| x - Function::single_val_matrix(2, 2, 1.), 1., -9.4);
    test!(sub_matrix_l_test,
          |x: Function| Function::single_val_matrix(2, 2, 1.) - x, 1., -8.4);
    test!(complex_test, |x: Function| {
        let a = Function::scalar(100.);
        let b = Function::matrix(2, 2, vec![
                                8., -5.,
                                0., 10.
                                ]);
        let c = Function::scalar(10.);
        sq(((&x - &a) * (&x + &b) - c))
    }, 0.0001, 0.00001);

    #[test]
    fn dot_test() {
        init();
        let x = Function::param("m", Constant::single_val(vec![2, 2], START));
        let c = Function::single_val_matrix(2, 2, 3.); 
        let f1 = dot(&c, &x);
        let f2 = dot(&x, &c);
        for f in &[f1, f2] {
            f.minimize(&HashMap::new(), 0.1, 10, 1);
            assert!(f.all_less_than(-29.), "F: {}", f);
        }
    }

    #[bench]
    fn run_lstm(b: &mut Bencher) {
        init();
        let dim = 10;
        let dims = vec![dim, dim];
        let inputs = vec![
            Constant::random(dims.clone(), -0.1, 0.1),
            Constant::random(dims.clone(), -0.1, 0.1)
            ]; 
        let target = Function::random_matrix(dims.clone(), -0.1, 0.1);
        let f = sq(lstm(inputs) - target);
        b.iter(|| {
            f.minimize(&HashMap::new(), 0.01, 100, 1000);
            println!("iterating...");
        });
    }
}
