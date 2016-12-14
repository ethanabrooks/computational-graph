extern {
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
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

impl Constant {
    pub fn width(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.width,
        }
    }

    pub fn height(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.height,
        }
    }
}
