use matrix::Matrix;
use rand::distributions::{IndependentSample, Range};
use rand;

#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

impl Constant {
    /// Returns `1` if `Constant` is a `Scalar`. Otherwise, returns the width of the `Matrix`.
    pub fn width(&self) -> usize {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.width(),
        }
    }

    /// Returns `1` if `Constant` is a `Scalar`. Otherwise, returns the height of the `Matrix`.
    pub fn height(&self) -> usize {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.height(),
        }
    }

    /// Creates a `Constant` with all values equal to `val`.
    /// # Examples
    /// ```
    /// assert_eq!(Constant::single_val(vec![], 2), Constant::Scalar(2))
    /// let x = Constant::single_val(vec![2, 2], 4)
    /// // x is the matrix [[4, 4],
    /// //                  [4, 4]]
    /// ```
    pub fn single_val(dims: Vec<usize>, val: f32) -> Constant {
        match dims.len() {
            0 => Constant::Scalar(val),
            2 => Constant::Matrix(Matrix::single_val(dims[0], dims[1], val)),
            _ => panic!("not supported"),
        }
    }

    /// Creates a `Constant` with values uniformly sampled between `lo` and `hi`.
    pub fn random(dims: Vec<usize>, lo: f32, hi: f32) -> Constant {
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

    /// Creates a `Constant::Matrix` with values drawn from `vals`. Note that although matrices
    /// store values in column-major order on the GPU, `vals` should take values in row-major
    /// order.
    ///
    /// # Example
    /// ```
    /// let x = Constant::matrix(2, 3, vec![1, 2, 3,
    ///                                     4, 5, 6])
    /// // x == [[1, 2, 3],
    /// //       [4, 5, 6]]
    /// ```
    pub fn matrix(height: usize, width: usize, vals: Vec<f32>) -> Constant {
        Constant::Matrix(Matrix::new(height as usize, width as usize, vals))
    }

    /// Allocates a new matrix (on the GPU if available) with dimensions identical to `self`, but
    /// with values all equal to `val`.
    /// # Example
    /// ```
    /// let x = Constant::matrix(2, 3, vec![1, 2, 3,
    ///                                     4, 5, 6])
    /// let y = x.copy_and_fill(4)
    /// // y == [[4, 4, 4],
    /// //       [4, 4, 4]]
    /// ```
    pub fn copy_and_fill(&self, val: f32) -> Constant {
        match *self {
            Constant::Scalar(_) => Constant::Scalar(val),
            Constant::Matrix(ref m) => 
                Constant::single_val(vec![m.height(), m.width()], val),
        }
    }


    pub fn copy(&mut self, other: &Constant) {
        match (&self, &other) {
            (&&mut Constant::Scalar(_), &&Constant::Matrix(_)) |
            (&&mut Constant::Matrix(_), &&Constant::Scalar(_)) => 
            panic!("Cannot copy mismatched Constants."),
            _ => {},
        }
        self.absorb(other);
    }

    pub fn absorb(&mut self, other: &Constant) {
        match (self, other) {
            (&mut Constant::Scalar(ref mut x1), &Constant::Scalar(x2)) => *x1 = x2,
            (&mut Constant::Matrix(ref mut m1), &Constant::Matrix(ref m2)) => m1.copy(m2),
            (&mut Constant::Scalar(ref mut x), &Constant::Matrix(_)) => *x = other.avg(),
            (&mut Constant::Matrix(ref mut m), &Constant::Scalar(ref x)) => m.fill(x.clone()),
        }
    }

    pub fn empty(dims: Vec<usize>) -> Constant {
        match &dims[..] {
            &[]              => Constant::Scalar(0.),
            &[height, width] => Constant::Matrix(Matrix::empty(height, width)),
            _                => panic!("not supported"),
        }
    }

    pub fn empty_like(c: &Constant) -> Constant {
        match *c {
            Constant::Scalar(_) => Constant::Scalar(0.),
            Constant::Matrix(ref m) => Constant::Matrix(Matrix::empty_like(m))
        }
    }

    pub fn empty_for_dot(c1: &Constant, trans1: bool, 
                         c2: &Constant, trans2: bool) -> Constant {
        match (c1, c2) {
            (&Constant::Scalar(_), &Constant::Scalar(_)) |
            (&Constant::Matrix(_), &Constant::Scalar(_)) |
            (&Constant::Scalar(_), &Constant::Matrix(_)) => Constant::empty_like(c1),
            (&Constant::Matrix(ref m1), &Constant::Matrix(ref m2)) => 
                Constant::Matrix(Matrix::empty_for_dot(m1, trans1, m2, trans2))
        }
    }

    pub fn dims(&self) -> Vec<usize> {
         match *self {
             Constant::Scalar(_) => vec![],
             Constant::Matrix(ref m) => m.dims(),
         }
    }
}
