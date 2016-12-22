use function::datatypes::{Function, Expr};
use std::collections::HashMap;
use std::ops::{Neg, Add, Sub, Mul};

macro_rules! fn1 {
    ($Op:ident, $op:ident) => {
        #[allow(dead_code)]
        pub fn $op(f: &Function) -> Function {
            Function::new(None, f.params.clone(), Expr::$Op(f.clone()))
        }
    };
}


macro_rules! apply2 {
    ($f1: expr, $f2:expr, $expr:expr) => {
        {
            let params1 = $f1.params.clone();
            let params2 = $f2.params.clone();
            let union = params1.union(&params2).cloned().collect();
            let function = Function::new(None, union, $expr);

            // optimization to combine constants
            match (&*$f1.body, &*$f2.body) { 
                (&Expr::Constant(_), &Expr::Constant(_)) =>
                    Function::constant(function.eval(&HashMap::new())),
                _ => function
            }
        }
    }
}

macro_rules! trait1 {
    ($Op:ident, $op:ident, $idem:expr) => {
        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self) -> Function {

                // optimization to eliminate identities
                if self.all_equal($idem) {
                    self.clone()
                } else {
                    Function::new(None, self.params.clone(), Expr::$Op(self.clone()))
                }
            }
        }

        impl $Op for Function {
            type Output = Function;
            fn  $op(self) -> Function { (&self).$op() }
        }
    }
}

macro_rules! trait2 {
    ($Op:ident, $op:ident, $identity:expr) => {
        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self, other: &Function) -> Function {
                let function = apply2!(self, other, Expr::$Op(self.clone(), other.clone()));

                // optimization to eliminate identities
                if self.all_equal($identity) {
                    other.clone()
                } else if other.all_equal($identity) {
                    self.clone()
                } else {
                    function
                }
            }
        }

        impl $Op for Function {
            type Output = Function;
            fn  $op(self, other: Function) -> Function { (&self).$op(&other) }
        }
    }
}

fn1!(Abs, abs);
fn1!(Sq, sq);
fn1!(Tanh, tanh);
fn1!(Signum, signum);
fn1!(Sigmoid, sigmoid);
trait1!(Neg, neg, 0.);
trait2!(Add, add, 0.);
trait2!(Sub, sub, 0.);
trait2!(Mul, mul, 0.);


impl Function {
    fn all_equal(&self, val:f32) -> bool {
        match *self.body {
            Expr::Constant(ref c) => c.all_equal(val),
            _                     => false
        }
    }

    pub fn signum(&self) -> Function { signum(&self) }
}

pub fn dot_transpose(f1: &Function, f2: &Function, trans1: bool, trans2: bool) -> Function {
    let function = apply2!(f1, f2, Expr::Dot(f1.clone(), f2.clone(), trans1, trans2));

    // optimization to combine constants
    match (&*f1.body, &*f2.body) { 
        (&Expr::Constant(_), &Expr::Constant(_)) =>
            Function::constant(function.eval(&HashMap::new())),
        _ => function
    }
}

pub fn dot(f1: &Function, f2: &Function) -> Function {
    dot_transpose(f1, f2, false, false)
}

