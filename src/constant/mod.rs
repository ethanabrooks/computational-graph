pub use self::datatypes::{Constant, Matrix};
pub use self::ops::{dot, mul_assign, add_assign, sub_assign, sigmoid_assign, 
                    signum_assign, abs_assign, negate, one_minus};

mod constructors;
mod datatypes;
mod ops;
mod print;

