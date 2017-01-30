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

    fn broadcast_sub_rev(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_mul_rev(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_add_rev(val: f32, m: *const Matrix, result: *mut Matrix);

    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_sub(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mul(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn gemm(m1: *const Matrix, trans1: bool, m2: *const Matrix, trans2: bool,
            result: *mut Matrix);
    fn all_equal(matrix: *const Matrix, x: f32) -> bool;
    fn all_less_than(matrix: *const Matrix, x: f32) -> bool;
    fn reduce_sum(matrix: *const Matrix) -> f32;
}

// constructor for single arg functions that preserve shape
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

macro_rules! Constant1 {
    ($op:ident, $c:expr, $scalar_fn:expr) => {{
        let matrix_fn = concat_idents!(map_, $op);
        match *$c {
            Constant::Scalar(ref mut x) => *x = $scalar_fn(x.clone()),
            Constant::Matrix(ref mut m) => unsafe { matrix_fn(m, m) }
        }
    }};
    ($op:ident, $result:expr, $arg:expr, $scalar_fn:expr) => {{
        let matrix_fn = concat_idents!(map_, $op);
        match ($result, $arg) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(x2)) => 
                *x1 = $scalar_fn(x2),
            (&mut Constant::Matrix(ref mut m), &Constant::Scalar(x)) => 
                m.fill($scalar_fn(x)),
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) => 
                unsafe { matrix_fn(m2, m1) },
            (&mut Constant::Scalar(ref mut x), &Constant::Matrix(_)) => 
                *x = $scalar_fn($arg.avg()),
        }
    }}
}

macro_rules! Constant2 {
    ($op:ident, $result:expr, $other:expr, $scalar_scalar_fn:expr) => {{

        // TODO: break out nonlinear form
        // Note: this function will resize values. If the operation is nonlinear, 
        // this may cause some unexpected results
        let matrix_scalar_fn = concat_idents!(broadcast_, $op);
        let matrix_matrix_fn = concat_idents!(elemwise_, $op);
        match ($result, $other) {
            (&mut Constant::Scalar(ref mut result), &Constant::Scalar(x)) => 
                *result = $scalar_scalar_fn(result.clone(), x),
            (&mut Constant::Matrix(ref mut result), &Constant::Scalar(x)) => 
                unsafe { matrix_scalar_fn(result, x, result) },
            (&mut Constant::Scalar(ref mut result), &Constant::Matrix(_)) => 
                *result = $scalar_scalar_fn(result.clone(), $other.avg()), // do not use with nonlinear functions
            (&mut Constant::Matrix(ref mut result), &Constant::Matrix(ref m)) => 
                unsafe { matrix_matrix_fn(result, m, result) },
        }
    }};

    ($op:ident, $result:expr, $c1:expr, $c2:expr, $scalar_scalar_fn:expr) => {{
        let matrix_scalar_fn = concat_idents!(broadcast_, $op);
        let scalar_matrix_fn = concat_idents!(broadcast_, $op, _rev);
        let matrix_matrix_fn = concat_idents!(elemwise_, $op);
        match ($result, $c1, $c2) {
            (&mut Constant::Scalar(ref mut result), 
                 &Constant::Scalar(x1), 
                 &Constant::Scalar(x2)) => *result = $scalar_scalar_fn(x1, x2),
            (&mut Constant::Matrix(ref mut result), 
                 &Constant::Scalar(x1), 
                 &Constant::Scalar(x2)) => result.fill($scalar_scalar_fn(x1, x2)),
            (&mut Constant::Matrix(ref mut result), 
                 &Constant::Matrix(ref m), 
                 &Constant::Scalar(x)) => unsafe { matrix_scalar_fn(m, x, result) },
            (&mut Constant::Matrix(ref mut result), 
                 &Constant::Scalar(x), 
                 &Constant::Matrix(ref m)) => unsafe { scalar_matrix_fn(x, m, result) },
            (&mut Constant::Matrix(ref mut result), 
                 &Constant::Matrix(ref m1), 
                 &Constant::Matrix(ref m2)) => unsafe { matrix_matrix_fn(m1, m2, result) },
                 _ => panic!("The result of this operation yields a matrix. Can't place 
                 a matrix result of a matrix operation in a scalar"),
        }
    }}
}

macro_rules! no_trait1 {

    // with identity
    ($Op:ident, $op:ident, $equals_op: ident,
     placeholders: $n_placeholders:expr, identity: $identity:expr) => {

        // use default implementation for scalar_fn
        no_trait1!($Op, $op, $equals_op, |x: f32| x.$op(),  // default implementation
                   placeholders: $n_placeholders, identity: $identity);
    };

    // with identity
    ($Op:ident, $op:ident, $equals_op: ident, $scalar_fn:expr, 
     placeholders: $n_placeholders:expr, identity: $identity:expr) => {

        impl Function {
            pub fn $op(&self) -> Function {

                // with identity
                Function1!($Op, $op, self, $n_placeholders, $identity)
            }
        }

        // allocates result
        impl Constant {
            pub fn $equals_op(&mut self, other: &Constant) {
                Constant1!($op, self, other, $scalar_fn);
            }

            pub fn $op(&self) -> Constant {
                let mut result = Constant::empty_like(self);
                Constant1!($op, &mut result, self, $scalar_fn);
                result
            }
        }
    };
}


macro_rules! Function2 {
    ($op:ident, $f1:expr, $f2:expr, 
     value:        $value:expr, 
     expr:         $expr:expr, 
     placeholders: $n_placeholders:expr) => {{
        // optimization to combine constants
        let params = $f1.params().clone()
                        .union(&($f2.params().clone()))
                        .cloned().collect();
        let function = Function::new($value, params, $expr, $n_placeholders);
        match ($f1.body(), $f2.body()) { 
            (&Expr::Constant(_), &Expr::Constant(_)) =>
                Function::constant(function.eval(&HashMap::new())),
            _ => function,
        }
    }};

    ($Op:ident, $op:ident, $f1:expr, $f2:expr, $n_placeholders:expr) => {{

        // value dims are equal to the bigger of the two args
        // TODO: can we calculate actual values here?
        let value = match (value!($f1), value!($f2)) {
            (&Constant::Scalar(ref x),  &Constant::Scalar(_)) => Constant::Scalar(x.clone()),
            (&Constant::Matrix(ref m),  &Constant::Scalar(_)) | 
            (&Constant::Scalar(_),      &Constant::Matrix(ref m)) => Constant::Matrix(m.clone()),
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
                assert!(m1.dims() == m2.dims());
                Constant::Matrix(m1.clone())
            }
        };
        let expr = Expr::$Op($f1.clone(), $f2.clone());
        Function2!($op, $f1, $f2, 
                   value:        value, 
                   expr:         expr, 
                   placeholders: $n_placeholders)
    }};

}


macro_rules! trait2 {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $equals_op:ident, 
     placeholders: $n_placeholders:expr, identity: $identity:expr) => {
        trait2!($Op, $op, $OpAssign, $op_assign, $equals_op, |x1: f32, x2:f32| x1.$op(x2),
                placeholders: $n_placeholders, identity: $identity);
    };

    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $equals_op:ident, 
     $scalar_scalar_fn:expr, placeholders: $n_placeholders:expr, identity: $identity:expr) => {

        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self, other: &Function) -> Function {

                // optimization to eliminate identities
                if self.all_equal($identity) {
                    other.clone()
                } else if other.all_equal($identity) {
                    self.clone()
                } else {
                    Function2!($Op, $op, self, other, $n_placeholders)
                }
            }
        }

        impl $Op for Function {
            type Output = Function;
            fn $op(self, other: Function) -> Function { (&self).$op(&other) }
        }

        impl<'a> $Op for &'a Constant {
            type Output = Constant;
            fn $op(self, other: &'a Constant) -> Constant {
                let mut result = Constant::empty_like(self);
                (&mut result).$op_assign(other);
                result
            }
        }

        impl $Op for Constant {
            type Output = Constant;
            fn $op(self, other: Constant) -> Constant { (&self).$op(&other) }
        }

        impl $OpAssign for Constant {

            fn $op_assign(&mut self, other: Constant) {
                self.$op_assign(&other);
            }
        }

        impl Constant {
            // Rust requires the argument for `_assign` methods in traits to not be refs.
            // This 'companion' method allows us to perform the same operation but 
            // using a ref as an argument
            pub fn $op_assign(&mut self, other: &Constant) {
                Constant2!($op, self, other, $scalar_scalar_fn)
            }

            // Like $op_assign in that it does not allocate, but instead of
            // performing the operation on self, it performs it on c1 and c2 and 
            // puts the result in self.
            pub fn $equals_op(&mut self, c1: &Constant, c2: &Constant) {
                Constant2!($op, self, c1, c2, $scalar_scalar_fn)
            }
        }
    };
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

no_trait1!(Abs, abs, equals_abs, placeholders: 1, identity: 0.);
no_trait1!(Signum, signum, equals_signum, placeholders: 0, identity: 0.);
no_trait1!(Sq, sq, equals_sq, |x: f32| x * x, placeholders: 0, identity: 0.);
no_trait1!(Tanh, tanh, equals_tanh, placeholders: 1, identity: 0.);
no_trait1!(Sigmoid, sigmoid, equals_sigmoid, |x: f32| 1. / (1. + (-x).exp()), 
           placeholders: 1, identity: 0.5);

trait2!(Add, add, AddAssign, add_assign, equals_add, 
        placeholders: 1, identity: 0.);
trait2!(Sub, sub, SubAssign, sub_assign, equals_sub, 
        placeholders: 1, identity: 0.);
trait2!(Mul, mul, MulAssign, mul_assign, equals_mul, 
        placeholders: 1, identity: 1.);

impl Constant {
    pub fn equals_one_minus(&mut self, other: &Constant) {
        Constant1!(one_minus, self, other, |x: f32| 1. - x);
    }

    pub fn assign_one_minus(&mut self) {
        Constant1!(one_minus, self, |x: f32| 1. - x);
    }

    pub fn equals_neg(&mut self, other: &Constant) {
        Constant1!(neg, self, other, |x: f32| -x);
    }

    pub fn assign_neg(&mut self) {
        Constant1!(neg, self, |x: f32| -x);
    }
}

impl<'a> Neg for &'a Function {
    type Output = Function;
    fn neg(self) -> Function {
        Function1!(Neg, neg, self, 0, 0.)
    }
}

// doesn't take a reference, but simply calls reference version
impl Neg for Function {
    type Output = Function;
    fn neg(self) -> Function {
        -(&self)
    }
}

// takes a reference, *allocates* result
impl<'a> Neg for &'a Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        let mut empty = Constant::empty_like(self);
        Constant1!(neg, &mut empty, self, |x: f32| -x);
        empty
    }
}

// doesn't take a reference, but simply calls reference version
impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant { -(&self) }
}

#[allow(dead_code)]
pub fn dot(f1: &Function, trans1: bool, f2: &Function, trans2: bool) -> Function {

    // optimization to combine constants
    match (f1.body(), f2.body()) { 
        (&Expr::Constant(ref c1), &Expr::Constant(ref c2)) =>
            Function::constant(Constant::dot(c1, trans1, c2, trans2)),
        _ => Function2!(dot, f1, f2, 
                        value:        Constant::empty_for_dot(value!(f1), trans1,
                                                              value!(f2), trans2),
                        expr:         Expr::Dot(f1.clone(), trans1, f2.clone(), trans2),
                        placeholders: 2),
    }
}

impl Constant {
    // TODO implement for scalars
    pub fn equals_dot(&mut self, c1: &Constant, trans1: bool, c2: &Constant, trans2: bool) {
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
        result.equals_dot(c1, trans1, c2, trans2);
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
