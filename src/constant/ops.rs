use constant::datatypes::{Constant, Matrix};
use std::ops::{Neg, Add, Sub, Mul, MulAssign, SubAssign};

extern {
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn map_sq(m: *const Matrix, result: *mut Matrix);
    fn map_abs(m: *const Matrix, result: *mut Matrix);
    fn map_signum(m: *const Matrix, result: *mut Matrix);
    fn map_sigmoid(m: *const Matrix, result: *mut Matrix);
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

    pub fn signum(&self) -> Constant {
        self.apply(&|x: f32| x.signum(), map_signum)
    }

    pub fn abs(&self) -> Constant {
        self.apply(&|x: f32| x.abs(), map_abs)
    }

    pub fn sigmoid(&self) -> Constant {
        self.apply(&sigmoid_f32, map_sigmoid)
    }

    pub fn sq(&self) -> Constant {
        self.apply(&|x| x * x, map_sq)
    }

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

//// FUNCTIONS

// TODO!!! trans does not work with matrix / scalar stuff!
// allocates on device
pub fn dot(c1: &Constant, c2: &Constant, trans1: bool, trans2: bool) -> Constant {
    let mut result;
    match (c1, c2) {
        (&Constant::Scalar(_), &Constant::Scalar(_)) =>
            panic!("dot should not be used for scalars"),
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) => {
            result = Matrix::empty_like(m);
            unsafe { broadcast_mul(x, m, &mut result) }
        }
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            result = Matrix::empty_like(m);
            unsafe { broadcast_mul_rev(m, x, &mut result) }
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            result = Matrix::empty_for_dot(m1, m2, trans1, trans2);
            unsafe { gemm(m1, trans1, m2, trans2, &mut result) }
        }
    };
    Constant::Matrix(result)
}
