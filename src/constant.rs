use std::fmt;
use std::ops::Neg;
use std::ops::Add;

type Matrix = Vec<f32>;

#[derive(Debug)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

impl Clone for Constant {
    fn clone(&self) -> Constant { 
        match self {
            &Constant::Scalar(ref x) => Constant::Scalar(x.clone()),
            &Constant::Matrix(ref m) => Constant::Matrix(m.clone())
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Constant::Scalar(x) => write!(f, "{}", x),
            &Constant::Matrix(ref m) => write!(f, "{:?}", m),
        }
    }
}

fn new_vec(size: usize, val: f32) -> Vec<f32> {
    let mut vec = Vec::with_capacity(size);
    for _ in 0..size {
        vec.push(val);
    }
    return vec;
}

pub fn copy_and_fill(c: &Constant, val: f32) -> Constant {
    match c {
        &Constant::Scalar(_) => Constant::Scalar(val),
        &Constant::Matrix(ref m) => Constant::Matrix(new_vec(m.len(), val)),
    }
}

pub fn new_constant(vec: Vec<i32>, val: f32) -> Constant {
    match vec.len() {
        0 => Constant::Scalar(val),
        2 => Constant::Matrix(new_vec((vec[0] * vec[1]) as usize, val)),
        _ => panic!("not supported"),
    }
}


fn apply(f: &Fn(f32) -> f32, c: &Constant) -> Constant {
    match c.clone() {
        Constant::Scalar(x) => Constant::Scalar(f(x)),
        Constant::Matrix(m) => Constant::Matrix(
            m.iter() 
                .map(|&x| f(x)) 
                .collect::<Matrix>()
        )
    }
}

mod bin {
    use constant::Matrix;
    use constant::Constant;

    pub fn apply(f: &Fn(f32, f32) -> f32, c1: Constant, c2: Constant) -> Constant {
        match (c1, c2) {
            (Constant::Scalar(x1), Constant::Scalar(x2)) => 
                Constant::Scalar(f(x1, x2)),
            (Constant::Matrix(m1), Constant::Matrix(m2)) => 
                Constant::Matrix(
                    m1.iter()
                    .zip(m2.iter())
                    .map(|(&x1, &x2)| f(x1, x2)) 
                    .collect::<Matrix>()
                ),
            (Constant::Scalar(x), Constant::Matrix(m)) 
                | (Constant::Matrix(m), Constant::Scalar(x)) =>
                Constant::Matrix(
                    m.iter()
                    .map(|e| f(*e, x))
                    .collect::<Matrix>()
                ),

        }
    }
}

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        apply(&|x| -x, &self)
    }
}

impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant {
        bin::apply(&|x1, x2| x1 + x2, self, other)
    }
}

