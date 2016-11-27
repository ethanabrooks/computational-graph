use std::{fmt, ptr};
use std::ops::{Neg, Add, Mul};

#[derive(Clone)]
#[repr(C)]
struct Matrix<'a> {
    height: i32,
    width: i32,
    devArray: *mut f32,
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

fn new_vec(size: usize, val: f32) -> Vec<f32> {
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(val);
    }
    return vec;
}

fn dummy_matrix(height: i32, width: i32) -> Matrix {
    Matrix { 
        height: height,
        width: width,
        devArray: ptr::null(),
    };
}

fn new_matrix_(height: i32, width: i32, vec: &Vec<f32>) -> Matrix {
    let matrix = Matrix { 
        height: height,
        width: width,
        devArray: ptr::null(),
    };
    unsafe { new_matrix(matrix, vec.as_mut_ptr(), height, width) };
    matrix
}

pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match *c {
        Constant::Scalar(_) => Constant::Scalar(val),
        Constant::Matrix(ref src) => {
            let dst = dummy_matrix(src.height, src.width);
            unsafe { copy_matrix(&src, &dst) };
            dst
        }
}

pub fn new_constant(dims: &Vec<i32>, val: f32) -> Constant {
    match dims.len() {
        0 => Constant::Scalar(val),
        2 => {
            let matrix = &mut dummy_matrix(dims[0], dims[1]);
            unsafe { 
                new_empty_matrix(matrix, dims[0], dims[1]);
                fill_matrix(matrix, val);
            };
            matrix
        },
        _ => panic!("not supported"),
    }
}


//fn apply(f: &Fn(f32) -> f32, c: &Constant) -> Constant {
    //match c.clone() {
        //Constant::Scalar(x) => Constant::Scalar(f(x)),
        //Constant::Matrix(m) => Constant::Matrix(
            //m.iter() 
                //.map(|&x| f(x)) 
                //.collect::<Matrix>()
        //)
        //// TODO: CUBLAS integration
    //}
//}

//mod bin {
    //use constant::Matrix;
    //use constant::Constant;

    //pub fn apply(f: &Fn(f32, f32) -> f32, c1: Constant, c2: Constant) -> Constant {
        //match (c1, c2) {
            //(Constant::Scalar(x1), Constant::Scalar(x2)) => 
                //Constant::Scalar(f(x1, x2)),
            //(Constant::Matrix(m1), Constant::Matrix(m2)) => 
                //Constant::Matrix(
                    //m1.iter()
                    //.zip(m2.iter())
                    //.map(|(&x1, &x2)| f(x1, x2)) 
                    //.collect::<Matrix>()
                //),
        //// TODO: CUBLAS integration
            //(Constant::Scalar(x), Constant::Matrix(m)) 
                //| (Constant::Matrix(m), Constant::Scalar(x)) =>
                //Constant::Matrix(
                    //m.iter()
                    //.map(|e| f(*e, x))
                    //.collect::<Matrix>()
                //),
        //// TODO: CUBLAS integration
        //}
    //}
//}

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        match c.clone() {
            Constant::Matrix(m) => 
            Constant::Scalar(x) => 
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
