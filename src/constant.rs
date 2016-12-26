use matrix::Matrix;
use rand::distributions::{IndependentSample, Range};
use rand;

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
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
            2 => Constant::Matrix(Matrix::single_val(dims[0], dims[1], val)),
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
        Constant::Matrix(Matrix::new(height as u32, width as u32, vals))
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
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(ref x2)) =>
                *x1 = *x2,
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) => 
                m1.copy(m2),
            _ => panic!("Can't copy from mismatched constant type.")
        }
    }

    pub fn empty_like(c: &Constant) -> Constant {
        match *c {
            Constant::Scalar(_) => Constant::Scalar(0.),
            Constant::Matrix(ref m) => Constant::Matrix(Matrix::empty_like(m))
        }
    }

    pub fn empty_for_dot(c1: &Constant, c2: &Constant, 
                        trans1: bool, trans2: bool) -> Constant {
        match (c1, c2) {
            (&Constant::Scalar(_), &Constant::Scalar(_)) |
            (&Constant::Matrix(_), &Constant::Scalar(_)) |
            (&Constant::Scalar(_), &Constant::Matrix(_)) => Constant::empty_like(c1),
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => 
                Constant::Matrix(Matrix::empty_for_dot(m1, m2, trans1, trans2))
        }
    }
}
