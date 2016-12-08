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
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => new_constant(&vec![m.height, m.width], val),
        }
    }
}

// allocates on device
pub fn empty_matrix(height: i32, width: i32) -> Matrix {
    let mut matrix = Matrix { 
        height: height,
        width: width,
        dev_array: ptr::null_mut(),
    };
    println!("allocating");
    unsafe { alloc_matrix(&mut matrix, height, width) };
    matrix
}

// allocates on device
pub fn empty_like(m: &Matrix) -> Matrix { empty_matrix(m.height, m.width) }

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

