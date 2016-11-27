use std::{fmt, ptr};
use std::ops::{Neg, Add, Mul};

#[repr(C)]
struct Matrix<'a> {
    height: i32,
    width: i32,
    devArray: *mut f32,
}

impl Copy for Matrix {
    fn clone(&self) -> Matrix {
        let m = new_matrix(self.height, self.width);
        copy_matrix(&self, &m);
        m
    }
}

#[derive(Debug, Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Constant::Scalar(x) => write!(f, "{}", x),
            Constant::Matrix(ref m) => write!(f, "{:?}", m),
        }
    }
}

//fn new_vec(size: usize, val: f32) -> Vec<f32> {
    //let mut vec = Vec::with_capacity(size);
    //for _ in 0..size {
        //vec.push(val);
    //}
    //return vec;
//}

//fn dummy_matrix(height: i32, width: i32) -> Matrix {
    //Matrix { 
        //height: height,
        //width: width,
        //devArray: ptr::null(),
    //};
//}

fn new_matrix(height: i32, width: i32) -> Matrix {
    let matrix = Matrix { 
        height: height,
        width: width,
        devArray: ptr::null(),
    };
    unsafe { alloc_matrix(matrix, height, width) };
    matrix
}

pub fn new_constant(dims: &Vec<i32>, val: f32) -> Constant {
    match dims.len() {
        0 => Constant::Scalar(val),
        2 => {
            let matrix = new_matrix(dims[0], dims[1]);
            unsafe { fill_matrix(matrix, val) };
            matrix
        },
        _ => panic!("not supported"),
    }
}

pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match *c {
        Constant::Scalar(_) => Constant::Scalar(val),
        Constant::Matrix(ref m) => new_constant(vec![m.height, m.width], val),

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        match c.clone() {
            Constant::Scalar(x) => Constant::Scalar(-x),
            Constant::Matrix(src) => {
                let dst = src.clone();
                neg()
            }
        }
    }
}

impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant {
        bin::apply(&|x1, x2| x1 + x2, self, other)
    }
}

impl Mul for Constant {
    type Output = Constant;
    fn mul(self, other: Constant) -> Constant {
        bin::apply(&|x1, x2| x1 * x2, self, other)
    }
}
