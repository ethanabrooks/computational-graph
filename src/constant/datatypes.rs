use std::process::exit;

extern {
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn free_matrix(m: *mut Matrix);
}

#[repr(C)]
pub struct Matrix {
    pub height: u32,
    pub width: u32,
    pub dev_array: *mut f32,
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

impl Matrix {
    pub fn size(&self) -> u32 {
        self.height * self.width
    }
}


impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut m = Matrix::empty_like(self);
        unsafe { copy_matrix(self as *const Matrix, &mut m) };
        m
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { free_matrix(self as *mut Matrix) };
    }
} 
