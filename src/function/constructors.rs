use constant::Constant;
use std::collections::HashSet;
use function::datatypes::{shared, Input, Param, Expr, Function};
use std::rc::Rc;
use std::cell::RefCell;

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert(String::from($val)); )*
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

    pub fn constant(value: Constant) -> Function {
        Function::new(Some(value.clone()), HashSet::new(), Expr::Constant(value))
    }

    #[allow(dead_code)]
    pub fn input(s: &str, dims: Vec<u32>) -> Function {
        Function::new(None, hashset![s], 
                    Expr::Input(Input {
                        name: String::from(s),
                        dims: dims,
                    }))
    }

    #[allow(dead_code)]
    pub fn param(s: &str, value: Constant) -> Function {
        Function::new(Some(value), 
                      hashset![s], 
                      Expr::Param(Param { name: String::from(s) }))
    }

    #[allow(dead_code)]
    pub fn random_param(s: &str, dims: Vec<u32>, lo: f32, hi: f32) -> Function {
        let value = Constant::random(dims, lo, hi);
        Function::new(Some(value), 
                      hashset![s], 
                      Expr::Param(Param { name: String::from(s) }))
    }

    #[allow(dead_code)]
    pub fn scalar(x: f32) -> Function {
        Function::constant(Constant::Scalar(x)) 
    }

    #[allow(dead_code)]
    pub fn matrix(height: u32, width: u32, values: Vec<f32>) -> Function {
        Function::constant(Constant::matrix(height, width, values))
    }

    #[allow(dead_code)]
    pub fn fill_matrix(height: u32, width: u32, value: f32) -> Function {
        Function::constant(Constant::single_val(vec![height, width], value))
    }
}

