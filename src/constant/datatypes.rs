#[repr(C)]
pub struct Matrix {
    pub height: i32,
    pub width: i32,
    pub dev_array: *mut f32,
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

impl Matrix {
    pub fn size(&self) -> i32 {
        self.height * self.width
    }
}

