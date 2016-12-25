use constant::Constant;
use std::cell::{RefCell, Ref, RefMut};
use std::collections::HashSet;
use std::rc::Rc;

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
    pub fn random_matrix(dims: Vec<u32>, lo: f32, hi: f32) -> Function {
        Function::constant(Constant::random(dims, lo, hi))
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
    pub fn single_val_matrix(height: u32, width: u32, value: f32) -> Function {
        Function::constant(Constant::single_val(vec![height, width], value))
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    value: Shared<Option<Constant>>,
    params: HashSet<String>,
    body: Rc<Expr>,
    placeholders: RefCell<Vec<Constant>>,
}

#[derive(Debug)]
pub enum Expr {
    Constant(Constant),
    Input(Input),
    Param(Param),
    Neg(Function),
    Sq(Function),
    Abs(Function),
    Signum(Function),
    Sigmoid(Function),
    Tanh(Function),
    Add(Function, Function),
    Sub(Function, Function),
    Mul(Function, Function),
    Dot(Function, Function, bool, bool),
}

#[derive(Debug)]
pub struct Input {
    dims: Vec<u32>, 
    pub name: String,
}

#[derive(Debug)]
pub struct Param {
    pub name: String,
}

type Shared<T> = Rc<RefCell<T>>;

pub mod shared {
    use super::Shared;
    use std::{cell, rc};

    pub fn new<T>(value: T) -> Shared<T> {
        rc::Rc::new(cell::RefCell::new(value))
    }
}

impl Function {
    pub fn body(&self) -> &Expr {
        &*self.body
    }

    pub fn params(&self) -> &HashSet<String> {
        &self.params
    }

    pub fn set_value(&self, value: Constant) {
        *(&self.value).borrow_mut() = Some(value);
    }

    pub fn get_value(&self) -> Ref<Option<Constant>> {
        self.value.borrow()
    }

   pub fn mutate_value(&self, f: &Fn(&mut Constant)) {
        match *self.value.borrow_mut() {
            Some(ref mut value) => f(value),
            None => panic!("Tried to mutate value that hasn't been assigned yet."),
        }
    }

    pub fn unwrap_value<'a>(&'a self) -> Ref<Constant> {
        Ref::map(self.value.borrow(), |x| match x.as_ref() {
            Some(x) => x,
            None => panic!("unwrap value failed on {:?}", self),
        })
    }

    pub fn unwrap_value_mut<'a>(&'a self) -> RefMut<Constant> {
        RefMut::map(self.value.borrow_mut(), |x| match *x {
            Some(ref mut x) => x,
            None => panic!("unwrap value failed on {:?}", self),
        })
    }

    pub fn alloc_placeholders(&self, c: Vec<Constant>) {
        *self.placeholders.borrow_mut() = c;
    }

    pub fn get_placeholder(&self, i: usize) -> RefMut<Constant> {
        RefMut::map(self.placeholders.borrow_mut(), |x| match x.get_mut(i) {
            Some(x) => x,
            None => panic!("Can't access placeholders[{}].", i),
        })
    }

    pub fn mutate_placeholder(&self, i: usize, f: &Fn(&mut Constant)) {
        match self.placeholders.borrow_mut().get_mut(i) {
            Some(ref mut placeholder) => f(placeholder),
            None => panic!("Tried to mutate a placeholder that hasn't been assigned yet."),
        }
    }

    pub fn num_placeholders(&self) -> usize {
        self.placeholders.borrow().len()
    }

    pub fn check_params(functions: Vec<&Function>) {
        for function in functions {
            match *function.body {
                Expr::Param(_) => {},
                _ => panic!("{} must be a param", &function),
            }
        }
    }
}
