#![feature(advanced_slice_patterns,
           slice_patterns,
           concat_idents,
           test,
           drop_types_in_const)]

extern crate rand;
extern crate test;

#[macro_use]
mod macros; 
mod function; 
mod constant; 
mod matrix; 
mod ops; 
mod optimize; 
mod print; 
mod lstm; 

use function::Function;
use std::collections::HashMap;

use constant::Constant;

fn main() {
    let args = HashMap::new();
    let dim = 10;
    let dims = vec![dim, dim];

    let x = Function::param("x", Constant::Scalar(0.5));
    //let a = Function::scalar(100.);
    //let b = Function::matrix(2, 2, vec![
                            //8., -5.,
                            //0., 10.
                            //]);
    //let c = Function::scalar(10.);
    let f = Function::single_val_matrix(2, 2, 1.) - x;
    //let f =((&x - &a) * (&x + &b) - c).sq();

    //let x = Function::param("x", Constant::Scalar(1.));
    //x.print();
    //let f = x.abs(); 
    //f.print();
    f.assign_values(&args);
    println!("f: ");
    f.print();
    f.value().print();

    //println!("g: ");
    //let g = &x + &b;
    //g.print();
    //println!("b: {}", b.value().avg());
    //println!("g: ");
    //g.assign_values(&args);
    //g.value().print();

    //let h = (&f * &g).sq();
    //println!("h: {:#?}", h);
    //h.assign_values(&args);
    //h.print();
    //h.value().print();

    //f.print();
    //f.backprop(&mut Constant::Scalar(1.), 0.1);
    //f.minimize(&args, 0.1, 10, 1);
    //let inputs = vec![
        //Constant::random(dims.clone(), -0.1, 0.1),
        //Constant::random(dims.clone(), -0.1, 0.1)
        //]; 
    //let target = Function::random_matrix(dims.clone(), -0.1, 0.1);
    //let f = sq(lstm(inputs) - target);
    //let m = Function::single_val_matrix(2, 2, 0.1);
    //let f = Function::random_param("x", dims.clone(), -0.1, 0.1);
    //f.minimize(&HashMap::new(), 0.01, 1000, 1000);
}


#[cfg(test)]
mod tests {
    use constant::Constant;
    use lstm::lstm;
    use super::*;
    use test::Bencher;
    use ops::dot;

    const START: f32 = 0.5;

    macro_rules! test {
        ($test_name:ident, $f:expr,
         learning rate: $learn_rate:expr, 
         test less than $goal:expr) => {
            #[test]
            fn $test_name () {
                let x = Function::param("x", Constant::Scalar(START));
                let m = Function::param("m", Constant::single_val(vec![2, 2], START));
                for f in &[$f(x), $f(m)] {
                    f.minimize(&HashMap::new(), $learn_rate, 10, 1);
                    assert!(f.all_less_than($goal), "F: {}", f);
                }
            }
        };
        ($test_name:ident, $f:expr, 
         constants value: $const_val:expr, 
         learning rate: $learn_rate:expr, 
         test less than $goal:expr) => {
            #[test]
            fn $test_name () {
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

    test!(single_param_test, |x| x, 
          learning rate: 1., 
          test less than -9.49);

    test!(neg_test, |x: Function| -x, 
          learning rate: 1., 
          test less than -9.49);

    test!(sq_test, |x: Function| x.sq(),
          learning rate: 0.1,
          test less than 0.01);

    test!(abs_test, |x: Function| x.abs(),
          learning rate: 0.1, 
          test less than 0.00001);

    test!(sigmoid_test, |x: Function| x.sigmoid(),
          learning rate: 1.,
          test less than -9.49);

    test!(tanh_test, |x: Function| x.tanh(),
          learning rate: 1., 
          test less than -0.97);

    test!(add_test, |x: Function, y: Function| x + y,
          constants value: 1., 
          learning rate: 1., 
          test less than -7.4);

    test!(mul_test, |x: Function, y: Function| x * y,
          constants value: 2., 
          learning rate: 0.1, 
          test less than -2.60);

    test!(sub_scalar_l_test, |x: Function| Function::scalar(1.) - x,
          learning rate: 1., 
          test less than -8.4);

    test!(sub_matrix_r_test,
          |x: Function| x - Function::single_val_matrix(2, 2, 1.),
          learning rate: 1., 
          test less than -9.4);

    test!(sub_matrix_l_test,
          |x: Function| Function::single_val_matrix(2, 2, 1.) - x,
          learning rate: 1.,
          test less than -80.4);

    //test!(complex_test, |x: Function| {
        //let a = Function::scalar(100.);
        //let b = Function::matrix(2, 2, vec![
                                //8., -5.,
                                //0., 10.
                                //]);
        //let c = Function::scalar(10.);
        ////let stuff = ((&((&x - &a) * (&x + &b)) - &c).sq());
        ////println!("\n\nVALUE");
        ////stuff.value().print();
        ////println!("VALUE\n\n");
        //(((&x - &a) * (&x + &b) - c)) // .sq())
    //}, 
          //learning rate: 0.00001,
          //test less than 0.0000);

    //#[test]
    //fn dot_test() {
        //let x = Function::param("m", Constant::single_val(vec![2, 2], START));
        //let c = Function::single_val_matrix(2, 2, 3.); 
        //let f1 = dot!(&c, &x);
        //let f2 = dot!(&x, &c);
        //for f in &[f1, f2] {
            //f.minimize(&HashMap::new(), 0.1, 10, 1);
            //assert!(f.all_less_than(-29.), "F: {}", f);
        //}
    //}

    #[bench]
    fn simple_op(b: &mut Bencher) {
        let f = Function::random_param("x", vec![10, 10], -0.1, 0.1);
        b.iter(|| {
            f.minimize(&HashMap::new(), 0.01, 1000, 10000);
            println!("iterating...");
        })
    }

    #[allow(dead_code)]
    fn run_lstm(b: &mut Bencher) {
        let dim = 10;
        let dims = vec![dim, dim];
        let inputs = vec![
            Constant::random(dims.clone(), -0.1, 0.1),
            Constant::random(dims.clone(), -0.1, 0.1)
            ]; 
        let target = Function::random_matrix(dims.clone(), -0.1, 0.1);
        let f = (lstm(inputs) - target).sq();
        b.iter(|| {
            f.minimize(&HashMap::new(), 0.01, 100, 1000);
            println!("iterating...");
        });
    }
}
