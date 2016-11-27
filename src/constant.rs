use std::fmt;
use std::ops::{Neg, Add, Mul};

type Matrix = Vec<f32>;

#[derive(Clone)]
#[repr(C)]
struct cMatrix<'a> {
    width: i32,
    height: i32,
    devArray: &'a [f32],
    array: &'a [f32],
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

pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match *c {
        Constant::Scalar(_) => Constant::Scalar(val),
        Constant::Matrix(ref m) => Constant::Matrix(new_vec(m.len(), val)),
        // TODO: set_matrix
    }
}

pub fn new_constant(dims: &Vec<i32>, val: f32) -> Constant {
    match dims.len() {
        0 => Constant::Scalar(val),
        2 => {
            let m = Matrix { ... };
            unsafe { new_matrix(m) };
            m
          }
        // TODO: set_matrix
        _ => panic!("not supported"),
    }
}


fn apply(f: &Fn(f32) -> f32, c: &Constant) -> Constant {
    match c.clone() {
        Constant::Scalar(x) => Constant::Scalar(f(x)),
        Constant::Matrix(m) => Constant::Matrix(
            m.iter() 
                .map(|&x| f(x)) 
                .collect::<Matrix>()
        )
        // TODO: CUBLAS integration
    }
}

mod bin {
    use constant::Matrix;
    use constant::Constant;

    pub fn apply(f: &Fn(f32, f32) -> f32, c1: Constant, c2: Constant) -> Constant {
        match (c1, c2) {
            (Constant::Scalar(x1), Constant::Scalar(x2)) => 
                Constant::Scalar(f(x1, x2)),
            (Constant::Matrix(m1), Constant::Matrix(m2)) => 
                Constant::Matrix(
                    m1.iter()
                    .zip(m2.iter())
                    .map(|(&x1, &x2)| f(x1, x2)) 
                    .collect::<Matrix>()
                ),
        // TODO: CUBLAS integration
            (Constant::Scalar(x), Constant::Matrix(m)) 
                | (Constant::Matrix(m), Constant::Scalar(x)) =>
                Constant::Matrix(
                    m.iter()
                    .map(|e| f(*e, x))
                    .collect::<Matrix>()
                ),
        // TODO: CUBLAS integration
        }
    }
}

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        apply(&|x| -x, &self)
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
