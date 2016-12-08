pub use self::constructors::{input, param, scalar, matrix};
pub use self::ops::dot;

mod constructors;
mod datatypes;
mod ops;
mod print;

use std::collections::HashMap;
use std::cell::Ref;
use std::io::{Write, stderr};
use std::ops::Deref;
use constant;
use constant::Constant;
use self::datatypes::{Function, Expr};
use self::constructors::new_constant;

impl Function {
    pub fn eval(&self, args: &HashMap<&str, Constant>) -> Constant {
        match *self.body { 
            Expr::Constant(ref x) => x.clone(),
            Expr::Input(ref i) => {
                match args.get::<str>(&i.name) {
                    Some(val) => val.clone(),
                    None => panic!("`args` is missing {}. Content of `args`: \n{:#?}",
                                   &i.name, args), 
                }
            }
            Expr::Param(ref p)           => self.unwrap_value().clone(),
            Expr::Neg(ref f)             => -f.eval(args),
            Expr::Abs(ref f)             => f.eval(args).abs(),
            Expr::Signum(ref f)          => f.eval(args).signum(),
            Expr::Sigmoid(ref f)         => f.eval(args).sigmoid(),
            Expr::Add(ref f1, ref f2)    => f1.eval(args) + f2.eval(args),
            Expr::Sub(ref f1, ref f2)    => f1.eval(args) - f2.eval(args),
            Expr::Mul(ref f1, ref f2)    => f1.eval(args) * f2.eval(args),
            Expr::MatMul(ref f1, ref f2) => constant::dot(&f1.eval(args), false,
                                                             &f2.eval(args), false)
        }
    }

    pub fn grad(&self, param: &str) -> Function {
        if self.params.contains::<str>(&param) {
            match *self.body { 
                Expr::Neg(ref f)             => -f.grad(param),
                Expr::Abs(ref f)             => f.signum() * f.grad(param),
                Expr::Signum(ref f)          => panic!("signum is nondifferentiable"),
                Expr::Sigmoid(ref f)         =>
                    f.grad(param) * (self.clone() * (&scalar(1.) - self)),
                Expr::Add(ref f1, ref f2)    => f1.grad(param) + f2.grad(param),
                Expr::Sub(ref f1, ref f2)    => f1.grad(param) - f2.grad(param),
                Expr::Mul(ref f1, ref f2)    => &f1.grad(param) * f2 +
                                                &f2.grad(param) * f1,
                Expr::MatMul(ref f1, ref f2) => panic!("still figuring this one out..."),
                Expr::Param(ref p) => new_constant(self.unwrap_value()
                                                       .copy_and_fill(1.)),
                Expr::Constant(_) | Expr::Input(_) => panic!("should never reach here"),
            }
        } else {
            scalar(0.)
        }
    }

    #[allow(dead_code)]
    pub fn minimize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        for _ in 0..iters {
            self.assign_values(&args);
            self.backprop((*self.unwrap_value()).copy_and_fill(1.), learn_rate);
            println!("{}", self.unwrap_value().clone());
        }
    }

    #[allow(dead_code)]
    pub fn maximize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        self.abs().minimize(args, learn_rate, iters);
    }
}

impl Function {
    fn assign_and_apply<'a>(&self, //f: &Fn(&Constant, &Constant) -> Constant, 
                        args: &'a HashMap<&str, Constant>,
                        value: Constant,
                        f1: &'a Function, f2: &'a Function) {
        f1.assign_values(args);
        f2.assign_values(args);
        //self.set_value(Some(f(f1.unwrap_value().deref(), f2.unwrap_value().deref())));
        self.set_value(Some(value));
    }

    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        match *self.body { 
            Expr::Constant(_) | Expr::Param(_) => assert!(self.get_value().is_some(),
                "Constants and Params must always have a value"),
            Expr::Input(ref i) => self.set_value(args.get::<str>(&i.name).cloned()),
            Expr::Neg(ref f1) => {
                f1.assign_values(args);
                self.mutate_or_set_value(Some(-(f1.unwrap_value()).deref()),
                                         &|x| *x *= Constant::Scalar(-1.));
           }
            Expr::Abs(ref f1) => {
                f1.assign_values(args);
                self.set_value(Some(f1.unwrap_value().abs().clone()))
            }
            Expr::Signum(ref f1) => {
                writeln!(&mut stderr(), "WARN: Signum is non-differentiable.
                Running `backprop` on this function will cause an error");
                f1.assign_values(args);
                self.set_value(Some(f1.unwrap_value().signum().clone()))
            }
            Expr::Sigmoid(ref f1) => {
                f1.assign_values(args);
                self.set_value(Some(f1.unwrap_value().sigmoid().clone()))
            }
            Expr::Add(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                self.set_value(Some(f1.unwrap_value().deref() + f2.unwrap_value().deref()));
            }
            Expr::Sub(ref f1, ref f2) => {
                //f1.assign_values(args);
                //f2.assign_values(args);
                //self.set_value(Some(f1.unwrap_value().deref() - f2.unwrap_value().deref()));
                self.assign_and_apply(args, 
                                      f1.unwrap_value().deref() - f2.unwrap_value().deref(),
                                      f1, f2);
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                self.set_value(Some(f1.unwrap_value().deref() * f2.unwrap_value().deref()));
                //self.assign_and_apply(&|x, y| x * y, args, f1, f2);
            }
            Expr::MatMul(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                self.set_value(Some(constant::dot(f1.unwrap_value().deref(), false,
                                                  f2.unwrap_value().deref(), false)));
                //self.assign_and_apply(&|x, y| constant::dot(&x, false, &y, false), 
                                 //args, f1, f2)
            }
        }
    }


    // TODO: preallocate error at every layer
    fn backprop(&self, mut error: Constant, learn_rate: f32) {
        if self.params.is_empty() { return; }
        match *self.body.clone() {
            Expr::Param(ref p) => {
                error *= Constant::Scalar(learn_rate);
                self.mutate_value(&|x| x.sub_assign(&error));
            }
            Expr::Neg(ref f) => {
                error *= Constant::Scalar(-1.);
                f.backprop(error, learn_rate)
            }
            Expr::Abs(ref f) => f.backprop(f.unwrap_value()
                                            .signum() * error, learn_rate),
            Expr::Signum(ref f) => panic!("sign is not differentiable"),
            Expr::Sigmoid(ref f) => {
                let value = self.unwrap_value();
                error.mul_assign(value.deref());
                error *= &Constant::Scalar(1.) - value.deref(); // CLONE
                f.backprop(error, learn_rate)
            }
            Expr::Add(ref f1, ref f2) => {
                f1.backprop(error.clone(), learn_rate); // CLONE
                f2.backprop(error, learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                f2.backprop(-&error, learn_rate); // CLONE
                f1.backprop(error, learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.backprop(f2.unwrap_value().deref() * &error, learn_rate); // CLONE
                f2.backprop(f1.unwrap_value().deref() * &error, learn_rate); // CLONE
            }
            Expr::MatMul(ref f1, ref f2) => {
                let error1 = constant::dot(&error, false, &f2.unwrap_value(), true); // CLONE
                let error2 = constant::dot(&f1.unwrap_value(), true, &error, false); // CLONE
                f1.backprop(error1, learn_rate);
                f2.backprop(error2, learn_rate);
            }
            Expr::Constant(_)| Expr::Input(_) => return,
        }
    }
}
