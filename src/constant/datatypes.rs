
#[repr(C)]
pub struct Matrix {
    pub height: i32,
    pub width: i32,
    dev_array: *mut f32,
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

