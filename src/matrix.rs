use std::ptr;
use GPU;

extern {
    fn alloc_matrix(m: *mut Matrix, width: u32, height: u32);
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn download_matrix(m1: *const Matrix, m2: *mut f32);
    fn free_matrix(m: *mut Matrix);
    fn upload_matrix(array: *const f32, m: *mut Matrix);
    fn fill_matrix(m: *mut Matrix, value: f32);
}

#[repr(C)]
pub struct Matrix {
    height: u32,
    width: u32,
    dev_array: *mut f32,
}

impl Matrix {
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn array_ptr(&self) -> *const f32 {
        if *GPU {
            let ptr = Vec::with_capacity(self.size() as usize).as_mut_ptr();
            unsafe { download_matrix(self, ptr); }
            ptr
        } else {
            self.dev_array
        }
    }

    pub fn empty_like(m: &Matrix) -> Matrix { Matrix::empty(m.height(), m.width()) }

    pub fn new(height: u32, width: u32, mut values: Vec<f32>) -> Matrix {
        assert!(values.len() as u32 == height * width, "wrong number of values");
        let mut matrix: Matrix = Matrix::empty(height, width);
        if *GPU {
            unsafe { upload_matrix(values.as_ptr(), &mut matrix) };
        } else {
            matrix.dev_array = values.as_mut_ptr();
        }
        matrix
    }

    pub fn empty_for_dot(m1: &Matrix, m2: &Matrix, 
                         trans1: bool, trans2: bool) -> Matrix {
        if trans1 {
            if trans2 {
                Matrix::empty(m1.width().clone(), m2.height().clone())
            } else {
                Matrix::empty(m1.width().clone(), m2.width().clone())
            }
        } else {
            if trans2 {
                Matrix::empty(m1.height().clone(), m2.height().clone())
            } else {
                Matrix::empty(m1.height().clone(), m2.width().clone())
            }
        }
    }

    pub fn single_val(height: u32, width: u32, val: f32) -> Matrix {
        let mut matrix: Matrix = Matrix::empty(height, width);
        unsafe { fill_matrix(&mut matrix, val) };
        matrix
    }

    pub fn copy(&mut self, other: &Matrix) {
        unsafe { copy_matrix(other, self) }
    }

    pub fn size(&self) -> u32 {
        self.height * self.width
    }

    pub fn empty(height: u32, width: u32) -> Matrix {
        let mut matrix = Matrix { 
            height: height,
            width: width,
            dev_array: ptr::null_mut(),
        };
        unsafe { alloc_matrix(&mut matrix, height, width) };
        matrix
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut m: Matrix = Matrix::empty(self.height, self.width);
        unsafe { copy_matrix(self, &mut m) };
        m
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { free_matrix(self as *mut Matrix) };
    }
} 
