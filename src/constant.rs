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

