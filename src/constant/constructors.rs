use constant::datatypes::{Constant, Matrix};
use std::ptr;

extern {
    fn alloc_matrix(m: *mut Matrix, width: i32, height: i32); // allocates on device
    fn free_matrix(m: *mut Matrix);
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn init_matrix(m: *mut Matrix, array: *const f32, width: i32, height: i32);
    fn fill_matrix(m: *mut Matrix, value: f32);
}

impl Constant {
    // allocates on device
    pub fn new(dims: Vec<i32>, val: f32) -> Constant {
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

    // allocates on device
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => Constant::new(vec![m.height, m.width], val),
        }
    }

    pub fn empty_like(c: &Constant) -> Constant {
        match *c {
            Constant::Scalar(x) => Constant::Scalar(0.),
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
    pub fn empty(height: i32, width: i32) -> Matrix {
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
    pub fn new(height: i32, width: i32, values: Vec<f32>) -> Matrix {
        assert!(values.len() as i32 == height * width, "wrong number of values");
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



