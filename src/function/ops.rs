use function::datatypes::{Function, Expr};
use function::constructors::new_constant;
use std::collections::HashMap;
use std::ops::{Neg, Add, Sub, Mul};

fn apply2(expr: &Fn(Function, Function) -> Expr, 
             f1: &Function, f2: &Function, 
             identity: f32) -> Function {
    let params1 = f1.params.clone();
    let params2 = f2.params.clone();
    let union = params1.union(&params2).cloned().collect();
    let function = Function::new(None, union, expr(f1.clone(), f2.clone()));
    match (&*f1.body, &*f2.body) {
        (&Expr::Constant(_), &Expr::Constant(_)) =>

            // optimization to combine constants
            new_constant(function.eval(&HashMap::new())),
        _ => {

            // optimization to eliminate identities
            if f1.all_equal(identity) {
                f2.clone()
            } else if f2.all_equal(identity) {
                f1.clone()
            } else {
                function
            }
        }
    }
}

#[allow(dead_code)]
pub fn abs(f: &Function) -> Function {
    f.apply(&|f| Expr::Abs(f))
}

#[allow(dead_code)]
pub fn sigmoid(f: &Function) -> Function {
    f.apply(&|f| Expr::Sigmoid(f))
}

#[allow(dead_code)]
pub fn tanh(f: &Function) -> Function {
    f.apply(&|f| Expr::Tanh(f))
}

#[allow(dead_code)]
pub fn sq(f: &Function) -> Function {
    f.apply(&|f| Expr::Sq(f))
}

impl Function {
    fn apply(&self, expr: &Fn(Function) -> Expr) -> Function {
        Function::new(None, self.params.clone(), expr(self.clone()))
    }

    fn all_equal(&self, val:f32) -> bool {
        match *self.body {
            Expr::Constant(ref c) => c.all_equal(val),
            _                     => false
        }
    }

    pub fn signum(&self) -> Function {
        self.apply(&|f| Expr::Signum(f))
    }
}

impl Neg for Function {
    type Output = Function;
    fn neg(self) -> Function { -&self }
}

impl Add for Function {
    type Output = Function;
    fn add(self, other: Function) -> Function { &self + &other }
}

impl Sub for Function {
    type Output = Function;
    fn sub(self, other: Function) -> Function { &self - &other }
}

impl Mul for Function {
    type Output = Function;
    fn mul(self, other: Function) -> Function { &self * &other }
}

impl <'a> Neg for &'a Function {
    type Output = Function;
    fn neg(self) -> Function {

        // optimization to eliminate -0
        if self.all_equal(0.) {
            self.clone()
        } else {
            self.apply(&|f| Expr::Neg(f))
        }
    }
}

impl<'a> Add for &'a Function {
    type Output = Function;
    fn add(self, other: &Function) -> Function {
        apply2(&|f1, f2| Expr::Add(f1, f2), self, other, 0.) 
    }
}

impl<'a> Sub for &'a Function {
    type Output = Function;
    fn sub(self, other: &Function) -> Function {
        apply2(&|f1, f2| Expr::Sub(f1, f2), self, other, 0.) 
    }
}

impl<'a> Mul for &'a Function {
    type Output = Function;
    fn mul(self, other: &Function) -> Function {

        // optimization to eliminate multiplication by 0
        if self.all_equal(0.) {
            return self.clone()
        } 
        if other.all_equal(0.) {
            return other.clone()
        }
        apply2(&|f1, f2| Expr::Mul(f1, f2), self, other, 1.) 
    }
}

pub fn dot(f1: &Function, f2: &Function) -> Function {
    dot_transpose(f1, f2, false, false)
}

pub fn dot_transpose(f1: &Function, f2: &Function, trans1: bool, trans2: bool) -> Function {
    let params1 = f1.params.clone();
    let params2 = f2.params.clone();
    let union = params1.union(&params2).cloned().collect();
    let function = Function::new(None, union, 
                                 Expr::Dot(f1.clone(), f2.clone(), trans1, trans2));

    // optimization to combine constants
    match (&*f1.body, &*f2.body) { 
        (&Expr::Constant(_), &Expr::Constant(_)) =>
            new_constant(function.eval(&HashMap::new())),
        _ => function
    }
}

