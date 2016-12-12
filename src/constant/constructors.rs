use constant::datatypes::{Constant, Matrix};
use rand::distributions::{IndependentSample, Range};
use rand;
use std::ptr;
use libc;

extern {
    fn alloc_matrix(m: *mut Matrix, width: u32, height: u32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn init_matrix(m: *mut Matrix, array: *const f32, 
                   width: libc::c_uint, height: libc::c_uint);
    fn fill_matrix(m: *mut Matrix, value: f32);
}

impl Constant {
    // allocates on device
    pub fn new_single_val(dims: Vec<u32>, val: f32) -> Constant {
        match dims.len() {
            0 => Constant::Scalar(val),
            2 => {
                let mut matrix = Matrix::empty(dims[0], dims[1]);
                unsafe { fill_matrix(&mut matrix, val) };
                Constant::Matrix(matrix)
            },
            _ => panic!("not supported"),
        }
    }

    pub fn new_random(dims: Vec<u32>, lo: f32, hi: f32) -> Constant {
        let between = Range::new(lo, hi);
        let mut rng = rand::thread_rng();
        match dims.len() {
            0 => Constant::Scalar(between.ind_sample(&mut rng)),
            2 => {
                let len = dims[0] * dims[1];
                let mut vals = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    vals.push(between.ind_sample(&mut rng));
                }
                Constant::new_matrix(dims[0], dims[1], vals)
            },
            _ => panic!("not supported"),
        }
    }

    pub fn new_matrix(height: u32, width: u32, vals: Vec<f32>) -> Constant {
        Constant::Matrix(Matrix::new(height, width, vals))
    }

    // allocates on device
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => 
                Constant::new_single_val(vec![m.height, m.width], val),
        }
    }

    pub fn copy(&mut self, other: &Constant) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(ref x2)) => {
                *x1 = *x2;
            }
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) => 
                unsafe { copy_matrix(m2, m1) },
            _ => panic!("Can't copy from mismatched constant type.")
        }
    }

    pub fn empty_like(c: &Constant) -> Constant {
        match *c {
            Constant::Scalar(_) => Constant::Scalar(0.),
            Constant::Matrix(ref m) => Constant::Matrix(Matrix::empty_like(m))
        }
    }

    pub fn empty_for_dot(c1: &Constant, c2: &Constant, 
                        trans1: bool, trans2: bool) -> Constant {
        match (c1, c2) {
            (&Constant::Scalar(_), &Constant::Scalar(_)) |
            (&Constant::Matrix(_), &Constant::Scalar(_)) |
            (&Constant::Scalar(_), &Constant::Matrix(_)) => Constant::empty_like(c1),
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => 
                Constant::Matrix(Matrix::empty_for_dot(m1, m2, trans1, trans2))
        }
    }
}

impl Matrix {
    // allocates on device
    pub fn empty(height: u32, width: u32) -> Matrix {
        let mut matrix = Matrix { 
            height: height,
            width: width,
            dev_array: ptr::null_mut(),
        };
        //println!("allocating");
        unsafe { alloc_matrix(&mut matrix, height, width) };
        matrix
    }

    // allocates on device
    pub fn empty_like(m: &Matrix) -> Matrix { Matrix::empty(m.height, m.width) }

    // allocates on device
    pub fn new(height: u32, width: u32, values: Vec<f32>) -> Matrix {
        assert!(values.len() as u32 == height * width, "wrong number of values");
        let mut matrix = Matrix::empty(height, width);

        unsafe { init_matrix(&mut matrix, values.as_ptr(), height, width) };
        matrix
    }

    pub fn empty_for_dot(m1: &Matrix, m2: &Matrix, 
                        trans1: bool, trans2: bool) -> Matrix {
        if trans1 {
            if trans2 {
                Matrix::empty(m1.width.clone(), m2.height.clone())
            } else {
                Matrix::empty(m1.width.clone(), m2.width.clone())
            }
        } else {
            if trans2 {
                Matrix::empty(m1.height.clone(), m2.height.clone())
            } else {
                Matrix::empty(m1.height.clone(), m2.width.clone())
            }
        }
    }
}
