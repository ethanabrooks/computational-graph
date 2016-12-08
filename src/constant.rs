use std::{fmt, ptr};
use std::ops::{Neg, Add, Mul, Sub, SubAssign, MulAssign};

extern {
    fn alloc_matrix(m: *mut Matrix, width: i32, height: i32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn init_matrix(m: *mut Matrix, array: *const f32, width: i32, height: i32);
    fn fill_matrix(m: *mut Matrix, value: f32);
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn map_abs(m: *const Matrix, result: *mut Matrix);
    fn map_signum(m: *const Matrix, result: *mut Matrix);
    fn map_sigmoid(m: *const Matrix, result: *mut Matrix);
    fn broadcast_mul(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_add(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn broadcast_mul_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_sub(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mul(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn gemm(m1: *const Matrix, trans1: bool, m2: *const Matrix, trans2: bool,
            result: *mut Matrix);
    fn download_matrix(src: *const Matrix, dst: *mut f32);
    fn reduce_equal(matrix: *const Matrix, x: f32) -> bool;
    fn reduce_sum(matrix: *const Matrix) -> f32;
}

//// STURCTS AND ENUMS

#[repr(C)]
pub struct Matrix {
    height: i32,
    width: i32,
    dev_array: *mut f32,
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}


//// TRAITS

// allocates on device
impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let mut m = empty_like(self);
        unsafe { copy_matrix(self as *const Matrix, &mut m) };
        m
    }
}

fn size(matrix: &Matrix) -> i32 {
    matrix.height * matrix.width
}

fn fmt_(c: &Constant, f: &mut fmt::Formatter) -> fmt::Result {
    match *c {
        Constant::Scalar(x) => write!(f, "{}", x),
        Constant::Matrix(ref src) => {
            let mut dst = Vec::with_capacity(size(src) as usize);
            unsafe { download_matrix(src, dst.as_mut_ptr()) };
            let mut result;

            let h = src.height - 1;
            result = if h == 0 { write!(f, "\n{:^2}", "[") }
            else               { write!(f, "\n{:^2}", "⎡")
            };
            if result.is_err() { return result }

            for i in 0..src.height {

                for j in 0..src.width {
                    result = write!(f, "{:^10.3}", unsafe {
                        *dst.as_ptr().offset((i * src.width + j) as isize) 
                    });
                    if result.is_err() { return result }
                }

                result = if h == 0           { write!(f, "{:^2}\n", "]") }

                else     if i == 0 && h == 1 { write!(f, "{:^2}\n{:^2}", "⎤", "⎣" ) }

                else     if i == h - 1       { write!(f, "{:^2}\n{:^2}", "⎥", "⎣") }

                else     if i == 0           { write!(f, "{:^2}\n{:^2}", "⎤", "⎢") }

                else     if i == h           { write!(f, "{:^2}\n", "⎦") } 

                else                         { write!(f, "{:^2}\n{:^2}", "⎥", "⎢") };

                if result.is_err() { return result }
            }
            result
        }
    }
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}


fn bin_apply(scalar_fun: &Fn(f32, f32) -> f32, 
             scalar_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
             matrix_scalar_fun: unsafe extern "C" fn(*const Matrix, f32, *mut Matrix),
             matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
             c1: &Constant, c2: &Constant) -> Constant {
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) => {
            Constant::Scalar(scalar_fun(x1, x2)) }
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) => {
            let mut result = empty_like(m);
            unsafe { scalar_matrix_fun(x, m, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            let mut result = empty_like(m);
            unsafe { matrix_scalar_fun(m, x, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            let mut result = empty_like(m1);
            unsafe { matrix_fun(m1, m2, &mut result) };
            Constant::Matrix(result)
        }
    }
}



// allocates on device
fn bin_apply_comm(scalar_fun: &Fn(f32, f32) -> f32, 
             broadcast_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
             matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
             c1: &Constant, c2: &Constant) -> Constant {
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) => {
            Constant::Scalar(scalar_fun(x1, x2)) }
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) |
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            let mut result = empty_like(m);
            unsafe { broadcast_matrix_fun(x, m, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            let mut result = empty_like(m1);
            unsafe { matrix_fun(m1, m2, &mut result) };
            Constant::Matrix(result)
        }
    }
}

impl Constant {
    // allocates on device
    fn apply(&self, scalar_fun: &Fn(f32) -> f32, 
             matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix)
             ) -> Constant {
        match self {
            &Constant::Scalar(x) => Constant::Scalar(scalar_fun(x)),
            &Constant::Matrix(ref m) => {
                let mut result = empty_like(m);
                unsafe { matrix_fun(m, &mut result) };
                Constant::Matrix(result)
            }
        }
    }

    fn un_assign(&mut self, scalar_fun: &Fn(f32), 
                 matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix)) {
        match self {
            &mut Constant::Scalar(x) => scalar_fun(x),
            &mut Constant::Matrix(ref mut m) => unsafe { matrix_fun(m, m) },
        }
    }

    fn bin_assign(&mut self, other: &Constant, 
                  scalar_fun: &Fn(&mut f32, f32), 
                  matrix_scalar_fun: unsafe extern "C" fn(*const Matrix, f32, *mut Matrix),
                  matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix)) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(x2)) =>
                scalar_fun(x1, x2),
            (&mut Constant::Matrix(ref mut m), &Constant::Scalar(x)) =>
                unsafe { matrix_scalar_fun(m, x, m) },
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) =>
                unsafe { matrix_fun(m1, m2, m1) },
            (&mut Constant::Scalar(ref mut x), &Constant::Matrix(ref m)) =>
                panic!("Can't assign from a matrix to a scalar without doing weird stuff."),
        }
    }

    pub fn avg(&self) -> f32 {
        match self {
            &Constant::Scalar(x) => x,
            &Constant::Matrix(ref m) => {
                (unsafe { reduce_sum(m) }) / size(m) as f32
            }
        }
    }

    pub fn abs(&self) -> Constant {
        self.apply(&|x| x.abs(), map_abs)
    }

    pub fn signum(&self) -> Constant {
        self.apply(&|x| x.signum(), map_signum)
    }

    pub fn sigmoid(&self) -> Constant {
        self.apply(&|x| 1. / (1. + (-x).exp()), map_sigmoid)
    }

    // allocates on device
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => new_constant(&vec![m.height, m.width], val),
        }
    }

    pub fn all_equal(&self, val: f32) -> bool {
        match *self {
            Constant::Scalar(x) => x == val,
            Constant::Matrix(ref m) => unsafe { reduce_equal(m, val) },
        }
    }

    // TODO: use bin_assign
    pub fn assign_dot(&mut self, c1: &Constant, trans1: bool, c2: &Constant, trans2: bool) {
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
        self.apply(&|x| -x, map_neg)
    }
}

// allocates on device
impl<'a> Add for &'a Constant {
    type Output = Constant;
    fn add(self, other: &'a Constant) -> Constant {
        bin_apply_comm(&|x1, x2| x1 + x2,
                       broadcast_add,
                       elemwise_add,
                       &self, &other)
    }
}

// allocates on device
impl<'a> Sub for &'a Constant {
    type Output = Constant;
    fn sub(self, other: &'a Constant) -> Constant {
        bin_apply(&|x1, x2| x1 - x2,
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
        bin_apply_comm(&|x1, x2| x1 * x2,
                       broadcast_mul,
                       elemwise_mul,
                       &self, &other)
    }
}

impl SubAssign for Constant {
    fn sub_assign(&mut self, other: Constant) {
        self.bin_assign(&other, &|x1, x2| *x1 -= x2, broadcast_sub_rev, elemwise_sub);
    }
}

impl MulAssign for Constant {
    fn mul_assign(&mut self, other: Constant) {
        self.bin_assign(&other, &|x1, x2| *x1 *= x2, broadcast_mul_rev, elemwise_mul);
    }
}

impl Constant {
    pub fn mul_assign(&mut self, other: &Constant) {
        self.bin_assign(other, &|x1, x2| *x1 *= x2, broadcast_mul_rev, elemwise_mul);
    }
}

//// FUNCTIONS

// allocates on device
pub fn dot(c1: &Constant, trans1: bool, c2: &Constant, trans2: bool) -> Constant {
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) =>
            panic!("dot should not be used for scalars"),
        (&Constant::Scalar(x), &Constant::Matrix(ref m)) => {
            let mut result = empty_like(m);
            unsafe { broadcast_mul(x, m, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m), &Constant::Scalar(x)) => {
            let mut result = empty_like(m);
            unsafe { broadcast_mul_rev(m, x, &mut result) };
            Constant::Matrix(result)
        }
        (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => {
            let mut result = 
                if trans1 {
                    if trans2 {
                        empty_matrix(m1.width, m2.height)
                    } else {
                        empty_matrix(m1.width, m2.width)
                    }
                } else {
                    if trans2 {
                        empty_matrix(m1.height, m2.height)
                    } else {
                        empty_matrix(m1.height, m2.width)
                    }
                };
            unsafe { gemm(m1, trans1, m2, trans2, &mut result) };
            Constant::Matrix(result)
        }
    }
}

// allocates on device
fn empty_matrix(height: i32, width: i32) -> Matrix {
    let mut matrix = Matrix { 
        height: height,
        width: width,
        dev_array: ptr::null_mut(),
    };
    unsafe { alloc_matrix(&mut matrix, height, width) };
    matrix
}

// allocates on device
fn empty_like(m: &Matrix) -> Matrix { empty_matrix(m.height, m.width) }

// allocates on device
pub fn new_matrix(height: i32, width: i32, values: Vec<f32>) -> Constant {
    assert!(values.len() as i32 == height * width, "wrong number of values");
    let mut matrix = empty_matrix(height, width);
    unsafe { init_matrix(&mut matrix, values.as_ptr(), height, width) };
    Constant::Matrix(matrix)
}

// allocates on device
pub fn new_constant(dims: &Vec<i32>, val: f32) -> Constant {
    match dims.len() {
        0 => Constant::Scalar(val),
        2 => {
            let mut matrix = empty_matrix(dims[0], dims[1]);
            unsafe { fill_matrix(&mut matrix, val) };
            Constant::Matrix(matrix)
        },
        _ => panic!("not supported"),
    }
}

