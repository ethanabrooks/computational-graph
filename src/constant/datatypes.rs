
#[repr(C)]
pub struct Matrix {
    height: i32,
    width: i32,
    dev_array: *mut f32,
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

