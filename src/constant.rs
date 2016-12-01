use std::{fmt, ptr};
use std::ops::{Neg, Add, Mul, SubAssign};

extern {
    fn alloc_matrix(m: *mut Matrix, width: i32, height: i32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn init_matrix(m: *mut Matrix, array: *const f32, width: i32, height: i32);
    fn fill_matrix(m: *mut Matrix, value: f32);
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_sub(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mult(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn broadcast_mult(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_add(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_sub_rev(m: *const Matrix, val: f32, result: *mut Matrix);
    fn download_array(src: *const Matrix, dst: *mut f32);
}

#[repr(C)]
pub struct Matrix {
    height: i32,
    width: i32,
    dev_array: *mut f32,
}

// allocates on device
impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let mut m = empty_like(self);
        unsafe { copy_matrix(self as *const Matrix, &mut m) };
        m
    }
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

fn fmt_(c: &Constant, f: &mut fmt::Formatter) -> fmt::Result {
    match *c {
        Constant::Scalar(x) => write!(f, "{}", x),
        Constant::Matrix(ref src) => {
            let mut dst = Vec::with_capacity((src.height * src.width) as usize);
            unsafe { download_array(src, dst.as_mut_ptr()) };
            let mut result;
            result = write!(f, "\n");
            for i in 0..src.height {
                for j in 0..src.width {
                    result = write!(f, "{:7.0}", unsafe {
                        *dst.as_ptr().offset((i * src.width + j) as isize) 
                    });
                    if result.is_err() {
                        return result
                    }
                }
                result = write!(f, "\n");
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

// allocates on device
pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match *c {
        Constant::Scalar(_) => Constant::Scalar(val),
        Constant::Matrix(ref m) => new_constant(&vec![m.height, m.width], val),
    }
}

// allocates on device
fn un_apply(broadcast_fun: &Fn(f32) -> f32, 
            matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix),
            c: &Constant) -> Constant {
    match c {
        &Constant::Scalar(x) => Constant::Scalar(broadcast_fun(x)),
        &Constant::Matrix(ref m) => {
            let mut result = empty_like(m);
            unsafe { matrix_fun(m, &mut result) };
            Constant::Matrix(result)
        }
    }

}

// allocates on device
fn bin_apply(scalar_fun: &Fn(f32, f32) -> f32, 
             broadcast_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
             matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
             c1: &Constant, c2: &Constant) -> Constant {
    match (c1, c2) {
        (&Constant::Scalar(x1), &Constant::Scalar(x2)) =>
            Constant::Scalar(scalar_fun(x1, x2)),
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

// allocates on device
impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        un_apply(&|x| -x, map_neg, &self)
    }
}

// allocates on device
impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant {
        bin_apply(&|x1, x2| x1 + x2,
                  broadcast_add,
                  elemwise_add,
                  &self, &other)
    }
}

// allocates on device
impl Mul for Constant {
    type Output = Constant;
    fn mul(self, other: Constant) -> Constant {
        bin_apply(&|x1, x2| x1 * x2,
                  broadcast_mult,
                  elemwise_mult,
                  &self, &other)
    }
}

// allocates on device
impl<'a> Neg for &'a Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        un_apply(&|x| -x, map_neg, self)
    }
}

// allocates on device
impl<'a> Mul for &'a Constant {
    type Output = Constant;
    fn mul(self, other: &'a Constant) -> Constant { 
        bin_apply(&|x1, x2| x1 * x2,
                  broadcast_mult,
                  elemwise_mult,
                  &self, &other)
    }
}

impl SubAssign for Constant {
    fn sub_assign(&mut self, other: Constant) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), Constant::Scalar(x2)) => *x1 -= x2,
            (&mut Constant::Matrix(ref mut m), Constant::Scalar(x)) => {
               unsafe { broadcast_sub_rev(m, x, m) }
            }
            (&mut Constant::Matrix(ref mut m1), Constant::Matrix(ref m2)) => {
                unsafe { elemwise_sub(m1, m2, m1) }
            }
            _ => panic!("scalar -= matrix not implemented.")
        }
    }
}
