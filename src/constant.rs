use std::{fmt, ptr};
use std::ops::{Neg, Add, Mul};
use std::result::Result;

extern {
    fn alloc_matrix(m: *mut Matrix, width: i32, height: i32);
    fn init_matrix(m: *mut Matrix, array: *const f32, width: i32, height: i32);
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn fill_matrix(m: *mut Matrix, value: f32);
    fn map_neg(m: *const Matrix, result: *mut Matrix);
    fn elemwise_add(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn elemwise_mult(m1: *const Matrix, m2: *const Matrix, result: *mut Matrix);
    fn broadcast_mult(val: f32, m: *const Matrix, result: *mut Matrix);
    fn broadcast_add(val: f32, m: *const Matrix, result: *mut Matrix);
}

#[repr(C)]
pub struct Matrix {
    pub height: i32,
    pub width: i32,
    pub devArray: *mut f32,
}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let mut m = empty_matrix(self.height, self.width);
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
        Constant::Matrix(ref m) => {
            let mut result = Result::Ok(());
            for i in 0..m.height {
                for j in 0..m.width {
                    result = write!(f, "{:0.7}", unsafe { 
                        *m.devArray.offset((i * j + m.width) as isize)
                    });
                    if result.is_err() {
                        return result
                    }
                }
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

fn empty_matrix(height: i32, width: i32) -> Matrix {
    let mut matrix = Matrix { 
        height: height,
        width: width,
        devArray: ptr::null_mut(),
    };
    unsafe { alloc_matrix(&mut matrix, height, width) };
    matrix
}

pub fn new_matrix(height: i32, width: i32, values: Vec<f32>) -> Constant {
    assert!(values.len() as i32 == height * width, "wrong number of values");
    let mut matrix = empty_matrix(height, width);
    unsafe { init_matrix(&mut matrix, values.as_ptr(), height, width) };
    Constant::Matrix(matrix)
}

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

pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match *c {
        Constant::Scalar(_) => Constant::Scalar(val),
        Constant::Matrix(ref m) => new_constant(&vec![m.height, m.width], val),
    }
}

fn un_apply(broadcast_fun: &Fn(f32) -> f32, 
            matrix_fun: unsafe extern "C" fn(*const Matrix, *mut Matrix),
            c: Constant) -> Constant {
    match c {
        Constant::Scalar(x) => Constant::Scalar(broadcast_fun(x)),
        Constant::Matrix(src) => {
            let mut dst = src.clone();
            unsafe { matrix_fun(&dst, &mut dst) }
            Constant::Matrix(dst)
        }
    }

}

fn bin_apply(broadcast_fun: &Fn(f32, f32) -> f32, 
            broadcast_matrix_fun: unsafe extern "C" fn(f32, *const Matrix, *mut Matrix),
            matrix_fun: unsafe extern "C" fn(*const Matrix, *const Matrix, *mut Matrix),
            c1: Constant, c2: Constant) -> Constant {
    match (c1, c2) {
        (Constant::Scalar(x1), Constant::Scalar(x2)) =>
            Constant::Scalar(broadcast_fun(x1, x2)),
        (Constant::Scalar(x), Constant::Matrix(ref mut m)) |
        (Constant::Matrix(ref mut m), Constant::Scalar(x)) => {
            unsafe { broadcast_matrix_fun(x, m, m) };
            Constant::Matrix(m.clone())
        }
        (Constant::Matrix(ref mut m1), Constant::Matrix(ref m2)) => {
            unsafe { matrix_fun(m1, m2, m1) }
            Constant::Matrix(m1.clone())
        }
    }

}

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        un_apply(&|x| -x, map_neg, self)
    }
}

impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant {
        bin_apply(&|x1, x2| x1 + x2,
                   broadcast_add,
                   elemwise_add,
                   self, other)
    }
}

impl Mul for Constant {
    type Output = Constant;
    fn mul(self, other: Constant) -> Constant {
        bin_apply(&|x1, x2| x1 * x2,
                   broadcast_mult,
                   elemwise_mult,
                   self, other)
    }
}
