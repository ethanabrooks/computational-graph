pub use self::ops::{dot_transpose, dot, abs, sigmoid, sq, tanh, mul_assign, 
                    add_assign, sub_assign, sigmoid_assign, signum_assign, 
                    tanh_assign, sq_assign, abs_assign, negate, one_minus};
pub use self::datatypes::{Function, Expr, Constant};

mod constructors;
mod datatypes;
mod ops;
mod print;
mod optimize;
