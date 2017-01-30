use std::{ptr, slice};
use libc::{c_uint, c_float};

extern {
    fn alloc_matrix(m: *mut Matrix);
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn free_matrix(m: *mut Matrix);
    fn fill_matrix(m: *mut Matrix, value: c_float);
    fn get_array(m: *const Matrix, dst: *mut c_float);
    fn set_array(m: *mut Matrix, values: *const c_float, transpose: bool);
}

#[repr(C)]
pub struct Matrix {
    height: c_uint,
    width: c_uint,
    array: *mut c_float,
}

impl Matrix {
    pub fn height(&self) -> usize {
        self.height as usize
    }

    pub fn width(&self) -> usize {
        self.width as usize
    }

    pub fn dims(&self) -> Vec<usize> {
        vec![self.height(), self.width()]
    }

    pub fn size(&self) -> usize {
        self.height() * self.width()
    }

    pub fn to_slice<'a>(&self) -> &'a [c_float] {
        let ptr: *mut f32 = Vec::with_capacity(self.size()).as_mut_ptr();
        unsafe { 
            get_array(self, ptr);
            slice::from_raw_parts(ptr, self.size())
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.to_slice().to_vec()
    }

    pub fn empty(height: usize, width: usize) -> Matrix {
        let mut matrix = Matrix { 
            height: height as c_uint,
            width: width as c_uint,
            array: ptr::null_mut(),
        };
        unsafe { alloc_matrix(&mut matrix) };
        matrix
    }

    pub fn empty_like(m: &Matrix) -> Matrix { Matrix::empty(m.height(), m.width()) }

    pub fn new(height: usize, width: usize, values: Vec<f32>) -> Matrix {
        assert!(values.len() as usize == height * width, "wrong number of values");
        let mut matrix: Matrix = Matrix::empty(height, width);
        unsafe { set_array(&mut matrix, values.as_ptr(), true) };
        matrix
    }

    pub fn empty_for_dot(m1: &Matrix, trans1: bool,
                         m2: &Matrix, trans2: bool) -> Matrix {
        if trans1 {
            if trans2 {
                Matrix::empty(m1.width(), m2.height())
            } else {
                Matrix::empty(m1.width(), m2.width())
            }
        } else {
            if trans2 {
                Matrix::empty(m1.height(), m2.height())
            } else {
                Matrix::empty(m1.height(), m2.width())
            }
        }
    }

    pub fn fill(&mut self, val: f32) {
        unsafe { fill_matrix(self, val as c_float) };
    }

    pub fn single_val(height: usize, width: usize, val: f32) -> Matrix {
        let mut matrix: Matrix = Matrix::empty(height, width);
        matrix.fill(val);
        matrix
    }

    pub fn copy(&mut self, other: &Matrix) {
        unsafe { copy_matrix(other /* src */, self /* dst */) }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut m: Matrix = Matrix::empty_like(self);
        m.copy(self);
        m
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { free_matrix(self as *mut Matrix) };
    }
} 
