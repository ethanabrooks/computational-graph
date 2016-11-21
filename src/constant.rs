use std::ops::Neg;
use std::ops::Add;
use std::f32;

type Matrix = Vec<f32>;

#[derive(Debug)]
pub enum Constant {
	scalar(f32),
	matrix(Matrix)
}

impl Constant {
    fn apply(self, f: &Fn(f32) -> f32) -> Constant {
        match self {
            Constant::scalar(x) => Constant::scalar(f(x)),
            Constant::matrix(m) => Constant::matrix(
                m.iter() 
                 .map(|&x| f(x)) 
                 .collect::<Matrix>()
            )
        }
    }
}

mod bin {
    use constant::Matrix;
    use constant::Constant;
    use constant::Constant::scalar;
    use constant::Constant::matrix;

    pub fn apply(f: &Fn(f32, f32) -> f32, c1: Constant, c2: Constant) -> Constant {
        match (c1, c2) {
            (scalar(x1), scalar(x2)) => 
                scalar(f(x1, x2)),
            (matrix(m1), matrix(m2)) => 
                matrix(
                    m1.iter()
                    .zip(m2.iter())
                    .map(|(&x1, &x2)| f(x1, x2)) 
                    .collect::<Matrix>()
                ),
            _  => panic!("c1 and c2 must both be scalars or both be matrices")
        }
    }
}

impl Clone for Constant {
    fn clone(&self) -> Constant { 
        match self {
            &Constant::scalar(ref x) => Constant::scalar(x.clone()),
            &Constant::matrix(ref m) => Constant::matrix(m.clone())
        }
    }
}

impl Neg for Constant {
    type Output = Constant;
    fn neg(self) -> Constant {
        self.apply(&|x| -x)
    }
}

//pub fn tanh(constant: Constant) -> Constant {
    //constant.apply(&|x| x.tanh())
//}

impl Add for Constant {
    type Output = Constant;
    fn add(self, other: Constant) -> Constant {
        bin::apply(&|x1, x2| x1 + x2, self, other)
    }
}

//impl Sub for Constant {
    //type Output = Constant;
    //fn add(self, other: Constant) -> Constant {
        //bin::apply(&|x1, x2| x1 + x2, self, other)
    //}
//}

