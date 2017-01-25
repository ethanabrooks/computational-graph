use function::{Function, Expr};
use constant::Constant;
use matrix::Matrix;
use std::collections::HashMap;
use std::ops::Deref;
use std::ops::{Neg, Add, Sub, Mul, MulAssign, SubAssign, AddAssign};

extern {
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn map_sq(m: *const Matrix, result: *mut Matrix);
    fn map_abs(m: *const Matrix, result: *mut Matrix);
    fn map_signum(m: *const Matrix, result: *mut Matrix);
    fn map_sigmoid(m: *const Matrix, result: *mut Matrix);
    fn map_tanh(m: *const Matrix, result: *mut Matrix);
    fn map_one_minus(m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub(m: *const Matrix, val: f32, result: *mut Matrix);
    fn broadcast_mul(m: *const Matrix, val: f32, result: *mut Matrix);
    fn broadcast_add(m: *const Matrix, val: f32, result: *mut Matrix);
    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_sub(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mul(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn gemm(m1: *const Matrix, trans1: bool, m2: *const Matrix, trans2: bool,
            result: *mut Matrix);
    fn all_equal(matrix: *const Matrix, x: f32) -> bool;
    fn all_less_than(matrix: *const Matrix, x: f32) -> bool;
    fn reduce_sum(matrix: *const Matrix) -> f32;
}

// constructor for function
macro_rules! Function1 {
    ($Op:ident, $op:ident, $f:ident, $n_placeholders:expr) => {
        Function::new($f.value().clone(),
                        $f.params().clone(),
                        Expr::$Op($f.clone()),
                        $n_placeholders) 
    };

    ($Op:ident, $op:ident, $f:ident, $n_placeholders:expr, $identity:expr) => {{

        // optimization to eliminate identities
        if $f.all_equal($identity) {
            $f.clone()
        } else {
            Function1!($Op, $op, $f, $n_placeholders)
        }
    }}
}

macro_rules! Function2 {
    ($op:ident, $f1:expr, $f2:expr) => {
        // default implementation of $function
        Function2!($op, $f1, $f2, (Function::$op($f1.clone(), $f2.clone())))
    };

    ($op:ident, $f1:expr, $f2:expr, ($function:expr)) => {
        // optimization to combine constants
        match ($f1.body(), $f2.body()) { 
            (&Expr::Constant(_), &Expr::Constant(_)) =>
                Function::constant($function.eval(&HashMap::new())),
            _ => $function
        }
    }
}

macro_rules! mutateConstant1 {
    ($op:ident, $scalar_fn:expr) => {
        #[allow(dead_code)]
        pub fn $op(mutable: &mut Constant) {
            match *mutable {
                Constant::Scalar(ref mut x) => *x = $scalar_fn(*x),
                Constant::Matrix(ref mut m) => {
                    let matrix_fn = concat_idents!(map_, $op);
                    unsafe { matrix_fn(m, m) };
                }
            }
        }
    };
}

// implement op for constants
macro_rules! Constant1 {

    // allocates result
    ($self_:expr, $op:ident, $scalar_fn:expr) => {
        match *$self_ {
            Constant::Scalar(x) => Constant::Scalar($scalar_fn(x)),
            Constant::Matrix(ref m) => {
                let mut result: Matrix = Matrix::empty_like(m);
                let matrix_fn = concat_idents!(map_, $op);
                unsafe { matrix_fn(m, &mut result) };
                Constant::Matrix(result)
            }
        }
    }
}

macro_rules! no_trait1 {

    ($Op:ident, 
     $op:ident, 
     placeholders: $n_placeholders:expr) => {

        // use default implementation for scalar_fn
        no_trait1!($Op, $op, |x: f32| x.$op(),  // default implementation
             placeholders: $n_placeholders);
    };

    ($Op:ident, 
     $op:ident, 
     placeholders: $n_placeholders:expr, 
     identity: $identity:expr) => {

        // use default implementation for scalar_fn
        no_trait1!($Op, $op, |x: f32| x.$op(),  // default implementation
             placeholders: $n_placeholders, 
             identity: $identity);
    };

    // without identity
    ($Op:ident, 
     $op:ident,
     $scalar_fn:expr, 
     placeholders: 
     $n_placeholders:expr) => {

        // in-place mutation
        mutateConstant1!($op, $scalar_fn);

        impl Function {
            pub fn $op(&self) -> Function {

                // without identity
                Function1!($Op, $op, self, $n_placeholders)
            }
        }

        // allocates result
        impl Constant {
            pub fn $op(&mut self) -> Constant {
                Constant1!(self, $op, $scalar_fn)
            }
        }
    };

    // with identity
    ($Op:ident, 
     $op:ident, 
     $scalar_fn:expr, 
     placeholders: $n_placeholders:expr, 
     identity: $identity:expr) => {

        // in-place mutation
        mutateConstant1!($op, $scalar_fn);

        impl Function {
            pub fn $op(&self) -> Function {

                // with identity
                Function1!($Op, $op, self, $n_placeholders, $identity)
            }
        }

        // allocates result
        impl Constant {
            pub fn $op(&mut self) -> Constant {
                Constant1!(self, $op, $scalar_fn)
            }
        }
    }
}


macro_rules! trait1 {

    ($Op:ident, 
     $op:ident, 
     placeholders: $n_placeholders:expr, 
     identity: $identity:expr) => {

        // use default implementation for scalar_fn
        trait1!($Op, $op, |x: f32| x.$op(), 
             placeholders: $n_placeholders, 
             identity: $identity);
    };

    ($Op:ident, 
     $op:ident, 
     $scalar_fn:expr, 
     placeholders: $n_placeholders:expr, 
     identity: $identity:expr) => {

        // takes a reference
        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self) -> Function {
                Function1!($Op, $op, self, $n_placeholders, $identity)
            }
        }

        // doesn't take a reference, but simply calls reference version
        impl $Op for Function {
            type Output = Function;
            fn $op(self) -> Function {
                (&self).$op()
            }
        }

        // takes a reference, *allocates* result
        impl<'a> $Op for &'a Constant {
            type Output = Constant;
            fn $op(self) -> Constant {
                Constant1!(self, $op, $scalar_fn)
            }
        }

        // doesn't take a reference, but simply calls reference version
        impl $Op for Constant {
            type Output = Constant;
            fn $op(self) -> Constant { (&self).$op() }
        }

        // function applies mutation
        mutateConstant1!($op, $scalar_fn);
    }
}


macro_rules! trait2 {
    ($Op:ident, 
     $op:ident, 
     $OpAssign:ident, 
     $op_assign:ident, 
     placeholders: $n_placeholders:expr, 
     identity: $identity:expr) => {

        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self, other: &Function) -> Function {

                // optimization to eliminate identities
                if self.all_equal($identity) {
                    other.clone()
                } else if other.all_equal($identity) {
                    self.clone()
                } else {
                    Function2!($op, self, other)
                }
            }
        }

        impl $Op for Function {
            type Output = Function;
            //fn $op(self, other: Function) -> Function { (&self).$op(&other) }

            fn $op(self, other: Function) -> Function {
                let params1 = self.params().clone();
                let params2 = other.params().clone();
                let params = params1.union(&params2).cloned().collect();
                Function::new(self.value().clone(),
                              params,
                              Expr::$Op(self.clone(), other),
                              $n_placeholders) 
            }
        }

        impl<'a> $Op for &'a Constant {
            type Output = Constant;
            fn $op(self, other: &'a Constant) -> Constant {
                let mut result: Constant = Constant::empty_like(self);
                result.$op_assign(self, other);
                result
            }
        }

        impl $Op for Constant {
            type Output = Constant;
            fn $op(self, other: Constant) -> Constant { (&self).$op(&other) }
        }

        impl $OpAssign for Constant {
            fn $op_assign(&mut self, other: Constant) {
                self.$op(&other)
            }
        }

        impl Constant {
            pub fn $op(&mut self, other: &Constant) {
                let matrix_scalar_fun = concat_idents!(broadcast_, $op, );
                let matrix_fun = concat_idents!(elemwise_, $op);
                match (self, other) {
                    (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(x2)) => 
                        x1.$op_assign(x2),
                    (&mut Constant::Matrix(ref mut m), &Constant::Scalar(x)) =>
                        unsafe { matrix_scalar_fun(m, x, m) },
                    (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) =>
                        unsafe { matrix_fun(m1, m2, m1) },
                    (&mut Constant::Scalar(ref mut x), &Constant::Matrix(_)) =>
                        x.$op_assign(other.avg())
                }
            }

            pub fn $op_assign(&mut self, arg1: &Constant, arg2: &Constant) {
                self.copy(arg1);
                self.$op(arg2);
            }
        }
    }
}

macro_rules! compare {
    ($cmp:ident, $scalar_cmp:ident) => {
        impl Function {
            pub fn $cmp(&self, val:f32) -> bool {
                self.value().deref().$cmp(val)
            }
        }
        impl Constant {
            pub fn $cmp(&self, val: f32) -> bool {
                match *self {
                    Constant::Scalar(x) => x.$scalar_cmp(&val),
                    Constant::Matrix(ref m) => unsafe { $cmp(m, val) },
                }
            }
        }
    }
}

compare!(all_equal, eq);
compare!(all_less_than, lt);

no_trait1!(Abs, abs, placeholders: 1, identity: 0.);
no_trait1!(Tanh, tanh, placeholders: 1, identity: 0.);
no_trait1!(Sigmoid, sigmoid, |x: f32| 1. / (1. + (-x).exp()), 
           placeholders: 1, identity: 0.5);
no_trait1!(Signum, signum, placeholders: 0, identity: 0.);
no_trait1!(Sq, sq, |x: f32| x * x, placeholders: 0, identity: 0.);

trait1!(Neg, neg, placeholders: 0, identity: 0.);

trait2!(Add, add, AddAssign, add_assign, placeholders: 1, identity: 0.);
trait2!(Sub, sub, SubAssign, sub_assign, placeholders: 1, identity: 0.);
trait2!(Mul, mul, MulAssign, mul_assign, placeholders: 1, identity: 1.);

mutateConstant1!(one_minus, |x: f32| 1. - x);

pub fn negate(c: &mut Constant) {
    *c *= Constant::Scalar(-1.);
}

#[allow(dead_code)]
pub fn dot(f1: &Function, trans1: bool, f2: &Function, trans2: bool) -> Function {

    // optimization to combine constants
    match (f1.body(), f2.body()) { 
        (&Expr::Constant(ref c1), &Expr::Constant(ref c2)) =>
            Function::constant(Constant::dot(c1, trans1, c2, trans2)),
        _ => Function2!(dot, f1, f2, (dot(f1, trans1, f2, trans2))),
    }
}

impl Constant {
    // TODO implement for scalars
    pub fn dot_assign(&mut self, c1: &Constant, trans1: bool, c2: &Constant, trans2: bool) {
        match (self, c1, c2) {
            (&mut Constant::Matrix(ref mut result),
                 &Constant::Matrix(ref m1), 
                 &Constant::Matrix(ref m2)) => {
                unsafe { gemm(m1, trans1, 
                              m2, trans2, 
                              result) }
            }
            _ => panic!("dot should not be used with scalars"),
        };
    }

    pub fn dot(c1: &Constant, trans1: bool, c2: &Constant, trans2: bool) -> Constant {
        let mut result: Constant = Constant::empty_for_dot(c1, trans1, c2, trans2);
        result.dot_assign(c1, trans1, c2, trans2);
        result
    }

    pub fn avg(&self) -> f32 {
        match self {
            &Constant::Scalar(x) => x,
            &Constant::Matrix(ref m) => {
                (unsafe { reduce_sum(m) }) / m.size() as f32
            }
        }
    }
}


