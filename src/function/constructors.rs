use constant::{Constant, new_matrix};
use std::collections::HashSet;
use function::datatypes::{shared, Input, Param, Expr, Function};
use std::rc::Rc;

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert($val); )*
         set
    }}
}

pub fn new_function(value: Option<Constant>, 
                params: HashSet<String>,
                body: Expr) -> Function {
    Function {
        value: shared::new(value),
        params: params,
        body: Rc::new(body),
    }
}

pub fn new_constant(value: Constant) -> Function {
    new_function(Some(value.clone()), HashSet::new(), Expr::Constant(value))
}

#[allow(dead_code)]
pub fn input(s: &str, dims: Vec<i32>) -> Function {
    let params = hashset![String::from(s)];
    new_function(None, params, 
                 Expr::Input(Input {
                     name: String::from(s),
                     dims: dims,
                 }))
}

#[allow(dead_code)]
pub fn param(s: &str, value: Constant) -> Function {
    let params = hashset![String::from(s)];
    new_function(Some(value), params, Expr::Param(Param { name: String::from(s) }))
}

#[allow(dead_code)]
pub fn scalar(x: f32) -> Function {
    new_constant(Constant::Scalar(x)) 
}

#[allow(dead_code)]
pub fn matrix(height: i32, width: i32, values: Vec<f32>) -> Function {
    new_constant(new_matrix(height, width, values))
}

