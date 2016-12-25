use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::ptr;

extern {
    fn alloc_matrix(m: *mut Matrix, width: u32, height: u32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn download_matrix(m1: *const Matrix, m2: *mut f32);
    fn free_matrix(m: *mut Matrix);
    fn upload_matrix(array: *const f32, m: *mut Matrix);
    fn fill_matrix(m: *mut Matrix, value: f32);
}

unsafe impl Send for Matrix {}

use std::sync::{Once, ONCE_INIT};

type PoolType = HashMap<(u32, u32), Vec<PMatrix>>;

static mut POOL: Option<Mutex<PoolType>> = None;
static INIT: Once = ONCE_INIT;

pub struct PMatrix {
    matrix: Option<Matrix>,
}

#[repr(C)]
pub struct Matrix {
    height: u32,
    width: u32,
    dev_array: *mut f32,
}

fn get_pool<'a>() -> MutexGuard<'a, PoolType> {
    unsafe {
        match POOL {
            Some(ref mutex) => mutex.lock().unwrap(),
            None            => {
                INIT.call_once(|| {
                    POOL = Some(Mutex::new(HashMap::new()));
                });
                get_pool()
            }
        }
    }
}

impl Drop for PMatrix {
    fn drop(&mut self) {
        if let Some(matrix) = self.matrix.take() {
            get_pool().entry((matrix.height, matrix.width))
                      .or_insert(vec![])
                      .push(PMatrix::from(matrix));
        }
    }
}

impl PMatrix {
    pub fn empty(height: u32, width: u32) -> PMatrix {
        match get_pool()
            .entry((height, width))
            .or_insert(vec![])
            .pop() {
                Some(pmatrix) => pmatrix,
                None => PMatrix::from(Matrix::empty(height, width))
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

    pub fn single_val(height: u32, width: u32, val: f32) -> PMatrix {
        let mut matrix: PMatrix = PMatrix::empty(height, width);
        unsafe { fill_matrix(matrix.borrow_mut(), val) };
        matrix
    }

    pub fn copy(&mut self, other: &PMatrix) {
        unsafe { copy_matrix(other.borrow(), self.borrow_mut()) }
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
