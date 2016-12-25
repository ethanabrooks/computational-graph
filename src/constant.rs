use std::collections::HashMap;
use std::sync::Mutex;
use std::ptr;
use rand::distributions::{IndependentSample, Range};
use rand;

extern {
    fn alloc_matrix(m: *mut Matrix, width: u32, height: u32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn download_matrix(m1: *const Matrix, m2: *mut f32);
    fn free_matrix(m: *mut Matrix);
    fn upload_matrix(array: *const f32, m: *mut Matrix);
    fn fill_matrix(m: *mut Matrix, value: f32);
}

unsafe impl Send for Matrix {}

lazy_static! {
    static ref POOL: Mutex<HashMap<(u32, u32), Vec<PMatrix>>> = 
        Mutex::new(HashMap::new());
    //static ref CUDA_INIT: bool = false;
}

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(PMatrix)
}

pub struct PMatrix {
    matrix: Option<Matrix>,
}

#[repr(C)]
pub struct Matrix {
    height: u32,
    width: u32,
    dev_array: *mut f32,
}

impl Constant {
    pub fn width(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.width(),
        }
    }

    pub fn height(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.height(),
        }
    }

    pub fn single_val(dims: Vec<u32>, val: f32) -> Constant {
        match dims.len() {
            0 => Constant::Scalar(val),
            2 => {
                let mut matrix: PMatrix = PMatrix::empty(dims[0], dims[1]);
                unsafe { fill_matrix(matrix.borrow_mut(), val) };
                Constant::Matrix(matrix)
            },
            _ => panic!("not supported"),
        }
    }

    pub fn random(dims: Vec<u32>, lo: f32, hi: f32) -> Constant {
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
                Constant::matrix(dims[0], dims[1], vals)
            },
            _ => panic!("not supported"),
        }
    }

    pub fn matrix(height: u32, width: u32, vals: Vec<f32>) -> Constant {
        Constant::Matrix(PMatrix::new(height as u32, width as u32, vals))
    }

    // allocates on device
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => 
                Constant::single_val(vec![m.height(), m.width()], val),
        }
    }

    pub fn copy(&mut self, other: &Constant) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(ref x2)) => {
                *x1 = *x2;
            }
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) => 
                unsafe { copy_matrix(m2.borrow(), m1.borrow_mut()) },
            _ => panic!("Can't copy from mismatched constant type.")
        }
    }

    pub fn empty_like(c: &Constant) -> Constant {
        match *c {
            Constant::Scalar(_) => Constant::Scalar(0.),
            Constant::Matrix(ref m) => Constant::Matrix(PMatrix::empty_like(m))
        }
    }

    pub fn empty_for_dot(c1: &Constant, c2: &Constant, 
                        trans1: bool, trans2: bool) -> Constant {
        match (c1, c2) {
            (&Constant::Scalar(_), &Constant::Scalar(_)) |
            (&Constant::Matrix(_), &Constant::Scalar(_)) |
            (&Constant::Scalar(_), &Constant::Matrix(_)) => Constant::empty_like(c1),
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => 
                Constant::Matrix(PMatrix::empty_for_dot(m1, m2, trans1, trans2))
        }
    }
}


impl Drop for PMatrix {
    fn drop(&mut self) {
        if let Some(matrix) = self.matrix.take() {
            let mut pool = POOL.lock().unwrap();
            let mut matrices = pool.entry((matrix.height, matrix.width))
                                   .or_insert(vec![]);
            matrices.push(PMatrix::from(matrix));
        }
    }
}

impl Clone for PMatrix {
    fn clone(&self) -> Self {
        let mut m: PMatrix = PMatrix::empty_like(self);
        unsafe { copy_matrix(self.borrow(), m.borrow_mut()) };
        m
    }
}

impl PMatrix {
    pub fn from(m: Matrix) -> PMatrix {
        let x = Some(m);
        let res = PMatrix { matrix: x };
        res
    }

    pub fn borrow_mut(&mut self) -> &mut Matrix {
        match self.matrix {
            Some(ref mut matrix) => matrix,
            None                 => panic!("For some reason, a PMatrix
                                            doesn't contain a matrix")
        }
    }

    pub fn borrow(&self) -> &Matrix {
        match self.matrix {
            Some(ref matrix) => matrix,
            None             => panic!("For some reason, a PMatrix
                                        doesn't contain a matrix")
        }
    }

    pub fn height(&self) -> u32 {
        self.borrow().height
    }

    pub fn width(&self) -> u32 {
        self.borrow().width
    }

    pub fn size(&self) -> usize {
        self.borrow().size() as usize
    }

    pub fn array_ptr(&self) -> *const f32 {
        let ptr = Vec::with_capacity(self.size()).as_mut_ptr();
        unsafe { 
            download_matrix(self.borrow(), ptr);
        }
        ptr
    }

    pub fn empty(height: u32, width: u32) -> PMatrix {
        POOL.lock().unwrap()
            .entry((height, width))
            .or_insert(vec![PMatrix::from(Matrix::empty(height, width))])
            .pop().unwrap()
    }

    pub fn empty_like(m: &PMatrix) -> PMatrix { PMatrix::empty(m.height(), m.width()) }

    pub fn new(height: u32, width: u32, values: Vec<f32>) -> PMatrix {
        assert!(values.len() as u32 == height * width, "wrong number of values");
        let mut matrix: PMatrix = PMatrix::empty(height, width);
        unsafe { upload_matrix(values.as_ptr(), matrix.borrow_mut()) };
        matrix
    }

    pub fn empty_for_dot(m1: &PMatrix, m2: &PMatrix, 
                         trans1: bool, trans2: bool) -> PMatrix {
        if trans1 {
            if trans2 {
                PMatrix::empty(m1.width().clone(), m2.height().clone())
            } else {
                PMatrix::empty(m1.width().clone(), m2.width().clone())
            }
        } else {
            if trans2 {
                PMatrix::empty(m1.height().clone(), m2.height().clone())
            } else {
                PMatrix::empty(m1.height().clone(), m2.width().clone())
            }
        }
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



impl Matrix {
    pub fn size(&self) -> u32 {
        self.height * self.width
    }

    // allocates on device
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

