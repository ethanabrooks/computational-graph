use function::datatypes::{shared, Function, Expr};
use function::constructors::{new_function, new_constant};
use constant::Constant;
use std::collections::HashMap;
use std::rc::Rc;
use std::ops::{Neg, Add, Sub, Mul};

fn bin_apply(expr: &Fn(Function, Function) -> Expr, 
             f1: &Function, f2: &Function, 
             identity: f32) -> Function {
    let params1 = f1.params.clone();
    let params2 = f2.params.clone();
    let union = params1.union(&params2).cloned().collect();
    let function = new_function(None, union, expr(f1.clone(), f2.clone()));
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


impl Function {
    fn apply(&self, expr: &Fn(Function) -> Expr) -> Function {
        Function {
            value: shared::new(None),
            params: self.params.clone(),
            body: Rc::new(expr(self.clone())),
        }
    }

    fn all_equal(&self, val:f32) -> bool {
        match *self.body {
            Expr::Constant(ref c) => c.all_equal(val),
            _                     => false
        }
    }

    // TODO: make methods assign in-place
    pub fn abs(&self) -> Function {
        self.apply(&|f| Expr::Abs(f))
    }

    pub fn signum(&self) -> Function {
        self.apply(&|f| Expr::Signum(f))
    }

    pub fn sigmoid(&self) -> Function {
        self.apply(&|f| Expr::Sigmoid(f))
    }
}

// TODO: macros!
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
        bin_apply(&|f1, f2| Expr::Add(f1, f2), self, other, 0.) 
    }
}

impl<'a> Sub for &'a Function {
    type Output = Function;
    fn sub(self, other: &Function) -> Function {
        bin_apply(&|f1, f2| Expr::Sub(f1, f2), self, other, 0.) 
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
        bin_apply(&|f1, f2| Expr::Mul(f1, f2), self, other, 1.) 
    }
}

// TODO: abstact some of this with bin_apply
// TODO: optimization for identities?
pub fn dot(f1: &Function, f2: &Function) -> Function {
    let params1 = f1.params.clone();
    let params2 = f2.params.clone();
    let union = params1.union(&params2).cloned().collect();
    let function = new_function(None, union, Expr::MatMul(f1.clone(), f2.clone()));

    // optimization to combine constants
    match (&*f1.body, &*f2.body) { 
        (&Expr::Constant(_), &Expr::Constant(_)) =>
            new_constant(function.eval(&HashMap::new())),
        _ => function
    }
}

