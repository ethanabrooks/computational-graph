pub use self::datatypes::Constant;
pub use self::constructors::new_matrix;
pub use self::ops::dot;

pub mod constructors;
mod datatypes;
mod ops;
mod traits;

