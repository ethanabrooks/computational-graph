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
use self::constructors::new_constant;
use self::datatypes::{Function, Expr};

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
            Expr::Dot(ref f1, ref f2) => constant::dot(&f1.eval(args), false,
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
                Expr::Dot(ref f1, ref f2) => panic!("still figuring this one out..."),
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
    fn rust_is_stupid<'a, 'b>(&self,
                        f1: &'a Function, f2: &'a Function,
                        args: &'a HashMap<&str, Constant>,
                        f: &Fn(&Constant, &Constant) -> Constant) {
        f1.assign_values(args);
        f2.assign_values(args);
        self.set_value(f(f1.unwrap_value().deref(), f2.unwrap_value().deref()));
    }

    fn recurse_and_set_value<'a, 'b>(&self,
                        f1: &'a Function, f2: &'a Function,
                        args: &'a HashMap<&str, Constant>,
                        value: Constant) {
        f1.assign_values(args);
        f2.assign_values(args);
        self.set_value(value);
    }

    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        let expr = *self.body;
        match expr { 
            Expr::Constant(_) | Expr::Param(_) => assert!(self.get_value().is_some(),
                "Constants and Params must always have a value"),
            Expr::Input(ref i) => {
                let arg = args.get::<str>(&i.name).expect("missing arg");
                self.set_value(arg.clone());
            }
            Expr::Neg(ref f1) => {
                f1.assign_values(args);
                self.maybe_allocate_for(expr);
                self.mutate_value(&|x| x.negate())
           }
            Expr::Abs(ref f1) => {
                f1.assign_values(args);
                self.set_value(f1.unwrap_value().abs().clone())
            }
            Expr::Signum(ref f1) => {
                writeln!(&mut stderr(), "WARN: Signum is non-differentiable.
                Running `backprop` on this function will cause an error");
                f1.assign_values(args);
                self.set_value(f1.unwrap_value().signum().clone())
            }
            Expr::Sigmoid(ref f1) => {
                f1.assign_values(args);
                self.set_value(f1.unwrap_value().sigmoid().clone())
            }
            Expr::Add(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                self.set_value(f1.unwrap_value().deref() + f2.unwrap_value().deref());
            }
            Expr::Sub(ref f1, ref f2) => {
                //self.rust_is_stupid(
                    //f1, f2, args,
                    //&|x: &Constant, y: &Constant| x - y)
                self.recurse_and_set_value(
                    f1, f2, args,
                    //&|x, y| x - y)
                    f1.unwrap_value().deref() - f2.unwrap_value().deref())
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                self.set_value(f1.unwrap_value().deref() * f2.unwrap_value().deref());
                //self.recurse_and_set_value(&|x, y| x * y, args, f1, f2);
            }
            Expr::Dot(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                let m1 = f1.unwrap_value();
                let m2 = f2.unwrap_value();
                self.maybe_allocate_for(expr);
                self.mutate_value(&|x| x.assign_dot(m1.deref(), m2.deref(), false, false));
            }
        }
    }

    fn maybe_allocate_for(&self, expr: Expr) {
        match *self.get_value() {
            Some(v) => return,
            None => match expr {
                Expr::Constant(_) | Expr::Input(_)| Expr::Param(_) => return,
                Expr::Neg(f) | Expr::Abs(f) | Expr::Signum(f) | Expr::Sigmoid(f) |
                Expr::Add(f, _) | Expr::Sub(f, _) | Expr::Mul(f, _) => 
                    self.set_value(Constant::empty_like(f.unwrap_value().deref())),
                Expr::Dot(f1, f2) => self.set_value(Constant::empty_for_dot(
                    f1.unwrap_value().deref(), f2.unwrap_value().deref(), false, false))
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
            Expr::Dot(ref f1, ref f2) => {
                let error1 = constant::dot(&error, false, &f2.unwrap_value(), true); // CLONE
                let error2 = constant::dot(&f1.unwrap_value(), true, &error, false); // CLONE
                f1.backprop(error1, learn_rate);
                f2.backprop(error2, learn_rate);
            }
            Expr::Constant(_)| Expr::Input(_) => return,
        }
    }
}
