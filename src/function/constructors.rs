use constant::{Constant, Matrix};
use std::collections::HashSet;
use function::datatypes::{shared, Input, Param, Expr, Function};
use std::rc::Rc;
use std::cell::RefCell;

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert($val); )*
         set
    }}
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
}

pub fn new_constant(value: Constant) -> Function {
    Function::new(Some(value.clone()), HashSet::new(), Expr::Constant(value))
}

#[allow(dead_code)]
pub fn input(s: &str, dims: Vec<i32>) -> Function {
    let params = hashset![String::from(s)];
    Function::new(None, params, 
                 Expr::Input(Input {
                     name: String::from(s),
                     dims: dims,
                 }))
}

#[allow(dead_code)]
pub fn param(s: &str, value: Constant) -> Function {
    let params = hashset![String::from(s)];
    Function::new(Some(value), params, Expr::Param(Param { name: String::from(s) }))
}

#[allow(dead_code)]
pub fn scalar(x: f32) -> Function {
    new_constant(Constant::Scalar(x)) 
}

#[allow(dead_code)]
pub fn matrix(height: i32, width: i32, values: Vec<f32>) -> Function {
    new_constant(Constant::Matrix(Matrix::new(height, width, values)))
}

#[allow(dead_code)]
pub fn fill_matrix(height: i32, width: i32, value: f32) -> Function {
    new_constant(Constant::new(vec![height, width], value))
}

