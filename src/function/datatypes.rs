use std::cell::{RefCell, Ref, RefMut};
use std::collections::HashSet;
use std::rc::Rc;

extern {
    fn copy_matrix(m1: *const Matrix, m2: *mut Matrix);
    fn free_matrix(m: *mut Matrix);
}

#[derive(Debug, Clone)]
pub struct Function {
    pub value: Shared<Option<Constant>>,
    pub params: HashSet<String>,
    pub body: Rc<Expr>,
    pub placeholders: RefCell<Vec<Constant>>,
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
    pub dims: Vec<u32>, 
    pub name: String,
}

#[derive(Debug)]
pub struct Param {
    pub name: String,
}


#[derive(Clone)]
pub enum Constant {
    Scalar(f32),
    Matrix(Matrix)
}

#[repr(C)]
pub struct Matrix {
    pub height: u32,
    pub width: u32,
    pub dev_array: *mut f32,
}

type Shared<T> = Rc<RefCell<T>>;

pub mod shared {
    use function::datatypes::Shared;
    use std::{cell, rc};

    pub fn new<T>(value: T) -> Shared<T> {
        rc::Rc::new(cell::RefCell::new(value))
    }
}

impl Function {
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
}


impl Matrix {
    pub fn size(&self) -> u32 {
        self.height * self.width
    }
}


impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut m = Matrix::empty_like(self);
        unsafe { copy_matrix(self as *const Matrix, &mut m) };
        m
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { free_matrix(self as *mut Matrix) };
    }
} 

impl Constant {
    pub fn width(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.width,
        }
    }

    pub fn height(&self) -> u32 {
        match *self {
            Constant::Scalar(_) => 1,
            Constant::Matrix(ref m) => m.height,
        }
    }
}
