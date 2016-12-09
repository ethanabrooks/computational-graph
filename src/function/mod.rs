pub use self::constructors::{input, param, scalar, matrix, new_constant};
pub use self::ops::dot;

mod constructors;
mod datatypes;
mod ops;
mod print;

use std::collections::HashMap;
use std::cell::Ref;
use std::io::{Write, stderr};
use std::ops::{Deref, DerefMut};
use constant;
use constant::Constant;
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
            Expr::Param(ref p)        => self.unwrap_value().clone(),
            Expr::Neg(ref f)          => -f.eval(args),
            Expr::Abs(ref f)          => f.eval(args).abs(),
            Expr::Signum(ref f)       => f.eval(args).signum(),
            Expr::Sigmoid(ref f)      => f.eval(args).sigmoid(),
            Expr::Add(ref f1, ref f2) => f1.eval(args) + f2.eval(args),
            Expr::Sub(ref f1, ref f2) => f1.eval(args) - f2.eval(args),
            Expr::Mul(ref f1, ref f2) => f1.eval(args) * f2.eval(args),
            Expr::Dot(ref f1, ref f2) => constant::dot(&f1.eval(args), &f2.eval(args), 
                                                       false, false)
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
            self.backprop(&mut self.unwrap_value().copy_and_fill(1.), learn_rate);
            //println!("{}", self.unwrap_value().clone());
        }
    }

    #[allow(dead_code)]
    pub fn maximize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        self.abs().minimize(args, learn_rate, iters);
    }
}

impl Function {
    fn recurse_and_set_value<'a, 'b>(&self,
                        f1: &'a Function, f2: &'a Function,
                        args: &'a HashMap<&str, Constant>,
                        value: Constant) {
        f1.assign_values(args);
        f2.assign_values(args);
        self.set_value(value);
    }

    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        self.assign_to_branches(args);
        self.maybe_alloc_value();
        match *self.body { 
            Expr::Constant(_) | Expr::Param(_) => assert!(self.get_value().is_some(),
                "Constants and Params must always have a value"),
            Expr::Input(ref i) => {
                let arg = args.get::<str>(&i.name).expect("missing arg");
                self.set_value(arg.clone());
            }
            Expr::Neg(ref f1) => {
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
                self.mutate_value(&|x| {
                    x.copy(f1.unwrap_value().deref());
                    x.add_assign(f2.unwrap_value().deref());
                })
                //f1.assign_values(args);
                //self.set_value(f1.unwrap_value().sigmoid().clone())
            }
            Expr::Add(ref f1, ref f2) => {
                self.mutate_value(&|x| {
                    x.copy(f1.unwrap_value().deref());
                    x.add_assign(f2.unwrap_value().deref());
                })
            }
            Expr::Sub(ref f1, ref f2) => {
                self.mutate_value(&|x| {
                    x.copy(f1.unwrap_value().deref());
                    x.sub_assign(f2.unwrap_value().deref());
                })
                //self.recurse_and_set_value(
                    //f1, f2, args,
                    ////&|x, y| x - y)
                    //f1.unwrap_value().deref() - f2.unwrap_value().deref())
            }
            Expr::Mul(ref f1, ref f2) => {
                self.mutate_value(&|x| {
                    x.copy(f1.unwrap_value().deref());
                    x.mul_assign(f2.unwrap_value().deref());
                })
                //self.set_value(f1.unwrap_value().deref() * f2.unwrap_value().deref());
                //self.recurse_and_set_value(&|x, y| x * y, args, f1, f2);
            }
            Expr::Dot(ref f1, ref f2) => {
                self.mutate_value(&|x| x.assign_dot(f1.unwrap_value().deref(), 
                                                    f2.unwrap_value().deref(), 
                                                    false, false));
                //self.set_value(constant::dot(f1.unwrap_value().deref(), 
                                             //f2.unwrap_value().deref(), 
                                             //false, false));
            }
        }
    }

    fn assign_to_branches(&self, args: &HashMap<&str, Constant>) {
        match *self.body {
            Expr::Constant(_)   | Expr::Input(_)   | Expr::Param(_) => return,
            Expr::Neg(ref f)    | Expr::Abs(ref f) | 
            Expr::Signum(ref f) | Expr::Sigmoid(ref f) => f.assign_values(args),
            Expr::Add(ref f1, ref f2) | Expr::Sub(ref f1, ref f2) | 
            Expr::Mul(ref f1, ref f2) | Expr::Dot(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
            }
        }
    }

    fn maybe_alloc_value(&self) {
        let new_value = match *self.get_value() {
            Some(_) => return,
            None => match *self.body {
                Expr::Constant(_)    | Expr::Input(_)      | Expr::Param(_) => return,
                Expr::Neg(ref f)     | Expr::Abs(ref f)    | Expr::Signum(ref f) | 
                Expr::Sigmoid(ref f) | Expr::Add(ref f, _) | 
                Expr::Sub(ref f, _)  | Expr::Mul(ref f, _) => 
                   Constant::empty_like(f.unwrap_value().deref()),
                Expr::Dot(ref f1, ref f2) =>
                    Constant::empty_for_dot(f1.unwrap_value().deref(), 
                                            f2.unwrap_value().deref(), 
                                            false, false),
            }
        };
        self.set_value(new_value);
    }

    // TODO: preallocate error at every layer
    fn backprop(&self, error: &mut Constant, learn_rate: f32) {
        let ref expr = *self.body;
        if self.params.is_empty() { return; }
        match *expr {
            Expr::Param(ref p) => {
                *error *= Constant::Scalar(learn_rate);
                self.mutate_value(&|x| x.sub_assign(&error));
            }
            Expr::Neg(ref f) => {
                *error *= Constant::Scalar(-1.);
                f.backprop(error, learn_rate)
            }
            Expr::Abs(ref f) => f.backprop(&mut (&f.unwrap_value()
                                                   .signum() * error), learn_rate),
            Expr::Signum(ref f) => panic!("sign is not differentiable"),
            Expr::Sigmoid(ref f) => {
                let value = self.unwrap_value();
                error.mul_assign(value.deref());
                *error *= &Constant::Scalar(1.) - value.deref(); // CLONE
                f.backprop(error, learn_rate)
            }
            Expr::Add(ref f1, ref f2) => {
                f1.backprop(&mut error.clone(), learn_rate); // CLONE
                f2.backprop(error, learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                f2.backprop(&mut -error.clone(), learn_rate); // CLONE
                f1.backprop(error, learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.backprop(&mut (f2.unwrap_value().deref() * &error), learn_rate);
                f2.backprop(&mut (f1.unwrap_value().deref() * &error), learn_rate);
            }
            Expr::Dot(ref f1, ref f2) => {
                //self.maybe_alloc_placeholders(expr, error);
                //self.mutate_placeholder(0, &|x| x.assign_dot(&error, &f2.unwrap_value(), 
                                                             //false, true));
                //self.mutate_placeholder(1, &|x| x.assign_dot(&f1.unwrap_value(), 
                                                             //&error, true, false));
                                                             
                //f1.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
                //f2.backprop(self.get_placeholder(1).deref_mut(), learn_rate);

                let mut error1 = constant::dot(&error, &f2.unwrap_value(), false, true);
                let mut error2 = constant::dot(&f1.unwrap_value(), &error, true, false);

                f1.backprop(&mut error1, learn_rate);
                f2.backprop(&mut error2, learn_rate);
            }
            Expr::Constant(_)| Expr::Input(_) => return,
        }
    }

    fn maybe_alloc_placeholders(&self, expr: &Expr, error: &Constant) {
        match *expr {
            Expr::Constant(_) | Expr::Input(_)   | Expr::Param(_) |
            Expr::Neg(_)  | Expr::Abs(_) | Expr::Signum(_) => return,
            Expr::Sigmoid(ref f) | Expr::Add(ref f, _) | 
            Expr::Sub(ref f, _)  | Expr::Mul(ref f, _) =>
                if self.num_placeholders() < 1 {
                    self.alloc_placeholders(
                        vec![Constant::empty_like(f.unwrap_value().deref())])
                },
            Expr::Dot(ref f1, ref f2) => 
                if self.num_placeholders() < 2 {
                    self.alloc_placeholders(
                        vec![Constant::empty_for_dot(
                                &error, &f2.unwrap_value().deref(), false, true),
                            Constant::empty_for_dot(
                                &f1.unwrap_value().deref(), &error, true, false)])
                }
        };
    }

}
