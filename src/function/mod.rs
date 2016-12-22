pub use self::datatypes::{Function, Expr, Constant};

mod constructors;
mod datatypes;
mod ops;
mod print;
mod optimize;

use function::datatypes::Matrix;
use std::collections::HashMap;
use std::ops::{Neg, Add, Sub, Mul, MulAssign, SubAssign};

extern {
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
    fn reduce_equal(matrix: *const Matrix, x: f32) -> bool;
    fn reduce_sum(matrix: *const Matrix) -> f32;
}


macro_rules! fn1 {
    ($Op:ident, $op:ident) => {
        fn1!($Op, $op, |x: f32| x.$op());
    };
    ($Op:ident, $op:ident, $scalar_fn:expr) => {
        pub fn $op(f: &Function) -> Function {
            Function::new(None, f.params.clone(), Expr::$Op(f.clone()))
        }
        impl Constant {
            pub fn $op(&self) -> Constant {
                match self {
                    &Constant::Scalar(x) => Constant::Scalar($scalar_fn(x)),
                    &Constant::Matrix(ref m) => {
                        let mut result = Matrix::empty_like(m);
                        unsafe { concat_idents!(map_, $op)(m, &mut result) };
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
fn1!(Tanh, tanh);
fn1!(Signum, signum);
fn1!(Sq, sq, |x: f32| x * x);
fn1!(Sigmoid, sigmoid, |x: f32| 1. / (1. + (-x).exp()));
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

fn apply2(scalar_fun: &Fn(f32, f32) -> f32, 
             scalar_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
             matrix_scalar_fun: unsafe extern "C" fn(*const Matrix, f32, *mut Matrix),
             matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
             c1: &Constant, c2: &Constant) -> Constant {
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) => {
            Constant::Scalar(scalar_fun(x1, x2)) }
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) => {
            let mut result = Matrix::empty_like(m);
            unsafe { scalar_matrix_fun(x, m, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            let mut result = Matrix::empty_like(m);
            unsafe { matrix_scalar_fun(m, x, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            let mut result = Matrix::empty_like(m1);
            unsafe { matrix_fun(m1, m2, &mut result) };
            Constant::Matrix(result)
        }
    }
}

// allocates on device
fn apply2_comm(scalar_fun: &Fn(f32, f32) -> f32, 
             broadcast_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
             matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
             c1: &Constant, c2: &Constant) -> Constant {
    let mut result;
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) => {
            Constant::Scalar(scalar_fun(x1, x2)) }
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) |
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            result = Matrix::empty_like(m);
            unsafe { broadcast_matrix_fun(x, m, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            result = Matrix::empty_like(m1);
            unsafe { matrix_fun(m1, m2, &mut result) };
            Constant::Matrix(result)
        }
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}


impl Constant {
    // allocates on device
    fn apply(&self, scalar_fun: &Fn(f32) -> f32, 
             matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix)
             ) -> Constant {
        match self {
            &Constant::Scalar(x) => Constant::Scalar(scalar_fun(x)),
            &Constant::Matrix(ref m) => {
                let mut result = Matrix::empty_like(m);
                unsafe { matrix_fun(m, &mut result) };
                Constant::Matrix(result)
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

    pub fn avg(&self) -> f32 {
        match self {
            &Constant::Scalar(x) => x,
            &Constant::Matrix(ref m) => {
                (unsafe { reduce_sum(m) }) / m.size() as f32
            }
        }
    }

    //pub fn sigmoid(&self) -> Constant {
        //self.apply(&sigmoid_f32, map_sigmoid)
    //}

    //pub fn sq(&self) -> Constant {
        //self.apply(&|x| x * x, map_sq)
    //}

    pub fn all_equal(&self, val: f32) -> bool {
        match *self {
            Constant::Scalar(x) => x == val,
            Constant::Matrix(ref m) => unsafe { reduce_equal(m, val) },
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
        let mut result;
        match (c1, c2) {
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
                result = Matrix::empty_for_dot(m1, m2, trans1, trans2);
                unsafe { gemm(m1, trans1, m2, trans2, &mut result) }
            }
            _ => panic!("dot should not be used with scalars"),
        };
        Constant::Matrix(result)
    }
}

// allocates on device
impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant { -&self }
}

// allocates on device
impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant { &self + &other }
}

// allocates on device
impl Sub for Constant {
    type Output = Constant;
    fn sub(self, other: Constant) -> Constant { &self - &other }
}

// allocates on device
impl Mul for Constant {
    type Output = Constant;
    fn mul(self, other: Constant) -> Constant { &self * &other }
}

// allocates on device
impl<'a> Neg for &'a Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        self.apply(&|x: f32| -x, map_neg)
    }
}

// allocates on device
impl<'a> Add for &'a Constant {
    type Output = Constant;
    fn add(self, other: &'a Constant) -> Constant {
        apply2_comm(&|x1: f32, x2| x1 + x2,
                       broadcast_add,
                       elemwise_add,
                       &self, &other)
    }
}

// allocates on device
impl<'a> Sub for &'a Constant {
    type Output = Constant;
    fn sub(self, other: &'a Constant) -> Constant {
        apply2(&|x1: f32, x2| x1 - x2,
                  broadcast_sub,
                  broadcast_sub_rev,
                  elemwise_sub,
                  &self, &other)
    }
}

// allocates on device
impl<'a> Mul for &'a Constant {
    type Output = Constant;
    fn mul(self, other: &'a Constant) -> Constant { 
        apply2_comm(&|x1, x2| x1 * x2,
                       broadcast_mul,
                       elemwise_mul,
                       &self, &other)
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
    c.assign1(&sigmoid_f32, map_sigmoid);
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

