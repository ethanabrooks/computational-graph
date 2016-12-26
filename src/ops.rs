use function::{Function, Expr};
use constant::Constant;
use matrix::Matrix;
use std::collections::HashMap;
use std::ops::{Neg, Add, Sub, Mul, MulAssign, SubAssign};

// TODO: wrappers
extern {
    fn init_cublas();
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn map_sq(m: *const Matrix, result: *mut Matrix);
    fn map_abs(m: *const Matrix, result: *mut Matrix);
    fn map_signum(m: *const Matrix, result: *mut Matrix);
    fn map_sigmoid(m: *const Matrix, result: *mut Matrix);
    fn map_tanh(m: *const Matrix, result: *mut Matrix);
    fn map_one_minus(m: *const Matrix, result: *mut Matrix);
    fn broadcast_mul(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_add(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn broadcast_mul_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn broadcast_add_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_sub(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mul(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn gemm(m1: *const Matrix, trans1: bool, m2: *const Matrix, trans2: bool,
            result: *mut Matrix);
    fn all_equal(matrix: *const Matrix, x: f32) -> bool;
    fn all_less_than(matrix: *const Matrix, x: f32) -> bool;
    fn reduce_sum(matrix: *const Matrix) -> f32;
}

pub fn init() {
    unsafe { init_cublas() };
}

macro_rules! fn1 {
    ($Op:ident, $op:ident, $op_ref:ident) => {
        fn1!($Op, $op, $op_ref, |x: f32| x.$op());
    };
    ($Op:ident, $op:ident, $op_ref:ident, $scalar_fn:expr) => {
        #[allow(dead_code)]
        pub fn $op(f: Function) -> Function {
            Function::new(None, f.params().clone(), Expr::$Op(f.clone()))
        }
        #[allow(dead_code)]
        pub fn $op_ref(f: &Function) -> Function {
            Function::new(None, f.params().clone(), Expr::$Op(f.clone()))
        }
        impl Constant {
            pub fn $op(&self) -> Constant {
                match self {
                    &Constant::Scalar(x) => Constant::Scalar($scalar_fn(x)),
                    &Constant::Matrix(ref m) => {
                        let mut result: Matrix = Matrix::empty_like(m);
                        let matrix_fn = concat_idents!(map_, $op);
                        unsafe { matrix_fn(m, &mut result) };
                        Constant::Matrix(result)
                    }
                }
            }
        }
    };
}


macro_rules! apply2 {
    ($f1: expr, $f2:expr, $expr:expr) => {
        {
            let params1 = $f1.params().clone();
            let params2 = $f2.params().clone();
            let union = params1.union(&params2).cloned().collect();
            let function = Function::new(None, union, $expr);

            // optimization to combine constants
            match ($f1.body(), $f2.body()) { 
                (&Expr::Constant(_), &Expr::Constant(_)) =>
                    Function::constant(function.eval(&HashMap::new())),
                _ => function
            }
        }
    }
}

macro_rules! trait1 {
    ($Op:ident, $op:ident, $idem:expr, $scalar_fn:expr) => {
        impl<'a> $Op for &'a Function {
            type Output = Function;
            fn $op(self) -> Function {

                // optimization to eliminate identities
                if self.all_equal($idem) {
                    self.clone()
                } else {
                    Function::new(None, self.params().clone(), Expr::$Op(self.clone()))
                }
            }
        }

        impl $Op for Function {
            type Output = Function;
            fn  $op(self) -> Function { (&self).$op() }
        }

        impl<'a> $Op for &'a Constant {
            type Output = Constant;
            fn $op(self) -> Constant {
                match self {
                    &Constant::Scalar(x) => Constant::Scalar($scalar_fn(x)),
                    &Constant::Matrix(ref m) => {
                        let mut result: Matrix = Matrix::empty_like(m);
                        let matrix_fn = concat_idents!(map_, $op);
                        unsafe { matrix_fn(m, &mut result) };
                        Constant::Matrix(result)
                    }
                }
            }
        }

        impl $Op for Constant {
            type Output = Constant;
            fn $op(self) -> Constant { (&self).$op() }
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

        impl<'a> $Op for &'a Constant {
            type Output = Constant;
            fn $op(self, other: &'a Constant) -> Constant {
                match (self, other) {
                    (&Constant::Scalar(x1), &Constant::Scalar(x2)) => {
                        Constant::Scalar(x1.$op(x2)) }
                    (&Constant::Scalar(x), &Constant::Matrix(ref m)) => {
                        let mut result: Matrix = Matrix::empty_like(m);
                        let scalar_matrix_fun = concat_idents!(broadcast_, $op);
                        unsafe { scalar_matrix_fun(x, m, &mut result) };
                        Constant::Matrix(result)
                    }
                    (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
                        let mut result = Matrix::empty_like(m);
                        let matrix_scalar_fun = concat_idents!(broadcast_, $op, _rev);
                        unsafe { matrix_scalar_fun(m, x, &mut result) };
                        Constant::Matrix(result)
                    }
                    (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
                        let mut result = Matrix::empty_like(m1);
                        let matrix_fun = concat_idents!(elemwise_, $op);
                        unsafe { matrix_fun(m1, 
                                            m2, 
                                            &mut result) };
                        Constant::Matrix(result)
                    }
                }
            }
        }

        impl $Op for Constant {
            type Output = Constant;
            fn $op(self, other: Constant) -> Constant { (&self).$op(&other) }
        }
    }
}

macro_rules! compare {
    ($cmp:ident, $scalar_cmp:ident) => {
        impl Function {
            pub fn $cmp(&self, val:f32) -> bool {
                match *self.get_value() {
                    Some(ref c) => c.$cmp(val),
                    _       => false
                }
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
//compare!(all_greater_than, gt);
fn1!(Abs, abs, abs_ref);
fn1!(Tanh, tanh, tanh_ref);
fn1!(Signum, signum, signum_ref);
fn1!(Sq, sq, sq_ref, |x: f32| x * x);
fn1!(Sigmoid, sigmoid, sigmoid_ref, |x: f32| 1. / (1. + (-x).exp()));
trait1!(Neg, neg, 0., |x: f32| -x);
trait2!(Add, add, 0.);
trait2!(Sub, sub, 0.);
trait2!(Mul, mul, 0.);

pub fn dot_transpose(f1: &Function, f2: &Function, trans1: bool, trans2: bool) -> Function {
    let function = apply2!(f1, f2, Expr::Dot(f1.clone(), f2.clone(), trans1, trans2));

    // optimization to combine constants
    match (f1.body(), f2.body()) { 
        (&Expr::Constant(_), &Expr::Constant(_)) =>
            Function::constant(function.eval(&HashMap::new())),
        _ => function
    }
}

pub fn dot(f1: &Function, f2: &Function) -> Function {
    dot_transpose(f1, f2, false, false)
}

impl Constant {
    pub fn avg(&self) -> f32 {
        match self {
            &Constant::Scalar(x) => x,
            &Constant::Matrix(ref m) => {
                (unsafe { reduce_sum(m) }) / m.size() as f32
            }
        }
    }

    fn assign1(&mut self, scalar_fun: &Fn(f32) -> f32, 
                 matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix)) {
        match self {
            &mut Constant::Scalar(x) => *self = Constant::Scalar(scalar_fun(x)),
            &mut Constant::Matrix(ref mut m) => unsafe { matrix_fun(m, m) },
        }
    }

    fn assign2(&mut self, other: &Constant, 
                  scalar_fun: &Fn(&mut f32, f32), 
                  matrix_scalar_fun: unsafe extern "C" fn(*const Matrix, f32, *mut Matrix),
                  matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix)) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(x2)) => scalar_fun(x1, x2),
            (&mut Constant::Matrix(ref mut m), &Constant::Scalar(x)) =>
                unsafe { matrix_scalar_fun(m, x, m) },
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) =>
                unsafe { matrix_fun(m1, m2, m1) },
            (&mut Constant::Scalar(ref mut x), &Constant::Matrix(_)) =>
                scalar_fun(x, other.avg())
        }
    }

    pub fn assign_dot(&mut self, c1: &Constant, c2: &Constant, trans1: bool, trans2: bool) {
        match (self, c1, c2) {
            (&mut Constant::Matrix(ref mut m), 
             &Constant::Scalar(x), &Constant::Matrix(ref m2)) => {
                unsafe { broadcast_mul(x, m2, m) };
            }
            (&mut Constant::Matrix(ref mut m),
            &Constant::Matrix(ref m1), &Constant::Scalar(x)) => {
                unsafe { broadcast_mul_rev(m1, x, m) };
            }
            (&mut Constant::Matrix(ref mut m),
            &Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
                unsafe { gemm(m1, trans1, m2, trans2, m) };
            }
            _ => panic!("Bad argument types for assign_dot")
        }
    }

    // allocates on device
    pub fn dot(c1: &Constant, c2: &Constant, trans1: bool, trans2: bool) -> Constant {
        let mut result: Matrix;
        match (c1, c2) {
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
                result = Matrix::empty_for_dot(m1, m2, trans1, trans2);
                unsafe { gemm(m1, trans1, 
                              m2, trans2, 
                              &mut result) }
            }
            _ => panic!("dot should not be used with scalars"),
        };
        Constant::Matrix(result)
    }
}

impl SubAssign for Constant {
    fn sub_assign(&mut self, other: Constant) {
        self.assign2(&other, &|x1: &mut f32, x2| *x1 -= x2, 
                      broadcast_sub_rev, elemwise_sub);
    }
}

impl MulAssign for Constant {
    fn mul_assign(&mut self, other: Constant) {
        self.assign2(&other, &|x1: &mut f32, x2| *x1 *= x2, 
                      broadcast_mul_rev, elemwise_mul);
    }
}

pub fn mul_assign(c: &mut Constant, other: &Constant) {
    c.assign2(other, &|x1: &mut f32, x2| *x1 *= x2,
                    broadcast_mul_rev, elemwise_mul);
}

pub fn add_assign(c: &mut Constant, other: &Constant) {
    c.assign2(other, &|x1: &mut f32, x2| *x1 += x2,
                    broadcast_add_rev, elemwise_add);
}

pub fn sub_assign(c: &mut Constant, other: &Constant) {
    c.assign2(other, &|x1: &mut f32, x2| *x1 -= x2,
                    broadcast_sub_rev, elemwise_sub);
}

pub fn sigmoid_assign(c: &mut Constant) {
    c.assign1(&|x| 1. / (1. + (-x).exp()), map_sigmoid);
}

pub fn tanh_assign(c: &mut Constant) {
    c.assign1(&|x| x.tanh(), map_tanh);
}

pub fn sq_assign(c: &mut Constant) {
    c.assign1(&|x| x * x, map_sq);
}

pub fn signum_assign(c: &mut Constant) {
    c.assign1(&|x| x.signum(), map_signum);
}

pub fn abs_assign(c: &mut Constant) {
    c.assign1(&|x| x.abs(), map_abs);
}

pub fn negate(c: &mut Constant) {
    *c *= Constant::Scalar(-1.)
}

pub fn one_minus(c: &mut Constant) {
    c.assign1(&|x| 1. - x, map_one_minus);
}

