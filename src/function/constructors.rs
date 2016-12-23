use std::collections::HashSet;
use function::datatypes::{shared, Input, Param, Expr, Function, Constant, PMatrix, Matrix};
use std::rc::Rc;
use std::ptr;
use std::cell::RefCell;
use rand::distributions::{IndependentSample, Range};
use rand;
use libc;

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert(String::from($val)); )*
         set
    }}
}

extern {
    fn alloc_matrix(m: *mut Matrix, width: u32, height: u32); // allocates on device
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn init_matrix(m: *mut Matrix, array: *const f32, 
                   width: libc::c_uint, height: libc::c_uint);
    fn fill_matrix(m: *mut Matrix, value: f32);
}


impl Function {
    pub fn new(value: Option<Constant>, 
               params: HashSet<String>,
               body: Expr) -> Function {
        Function {
            value: shared::new(value),
            params: params,
            body: Rc::new(body),
            placeholders: RefCell::new(vec![]),
        }
    }

    pub fn constant(value: Constant) -> Function {
        Function::new(Some(value.clone()), HashSet::new(), Expr::Constant(value))
    }

    #[allow(dead_code)]
    pub fn input(s: &str, dims: Vec<u32>) -> Function {
        Function::new(None, hashset![s], 
                    Expr::Input(Input {
                        name: String::from(s),
                        dims: dims,
                    }))
    }

    #[allow(dead_code)]
    pub fn param(s: &str, value: Constant) -> Function {
        Function::new(Some(value), 
                      hashset![s], 
                      Expr::Param(Param { name: String::from(s) }))
    }

    #[allow(dead_code)]
    pub fn random_param(s: &str, dims: Vec<u32>, lo: f32, hi: f32) -> Function {
        let value = Constant::random(dims, lo, hi);
        Function::new(Some(value), 
                      hashset![s], 
                      Expr::Param(Param { name: String::from(s) }))
    }

    #[allow(dead_code)]
    pub fn scalar(x: f32) -> Function {
        Function::constant(Constant::Scalar(x)) 
    }

    #[allow(dead_code)]
    pub fn matrix(height: u32, width: u32, values: Vec<f32>) -> Function {
        Function::constant(Constant::matrix(height, width, values))
    }

    #[allow(dead_code)]
    pub fn fill_matrix(height: u32, width: u32, value: f32) -> Function {
        Function::constant(Constant::single_val(vec![height, width], value))
    }
}

impl Constant {
    // allocates on device
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
        //assert!(height > 0 && width > 0);
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

impl Matrix {
    // allocates on device
    pub fn empty(height: u32, width: u32) -> Matrix {
        let mut matrix = Matrix { 
            height: height,
            width: width,
            dev_array: ptr::null_mut(),
        };
        //println!("allocating");
        unsafe { alloc_matrix(&mut matrix, height, width) };
        matrix
    }
}

impl PMatrix {
    // allocates on device
    pub fn empty(height: u32, width: u32) -> PMatrix {
        PMatrix::from(Matrix::empty(height, width))
    }

    // allocates on device
    pub fn empty_like(m: &PMatrix) -> PMatrix { PMatrix::empty(m.height(), m.width()) }

    // allocates on device
    pub fn new(height: u32, width: u32, values: Vec<f32>) -> PMatrix {
        assert!(values.len() as u32 == height * width, "wrong number of values");
        let mut matrix: PMatrix = PMatrix::empty(height, width);
        unsafe { init_matrix(matrix.borrow_mut(), values.as_ptr(), height, width) };
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
