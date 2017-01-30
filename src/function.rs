use constant::Constant;
use std::cell::{RefCell, Ref, RefMut};
use std::collections::HashSet;
use std::rc::Rc;
use std::ops::Index;

#[derive(Debug, Clone)]
struct Pool(Vec<RefCell<Constant>>);

#[derive(Debug, Clone)]
pub struct Function {
    value: RefCell<Constant>,
    params: HashSet<String>,
    body: Rc<Expr>,
    pool: Pool,
}

#[derive(Debug)]
pub enum Expr {
    Constant(Constant),
    //Input(Input),
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
    Dot(Function, bool, Function, bool),
}

//#[derive(Debug)]
//pub struct Input {
    //dims: Vec<usize>, 
    //pub name: String,
//}

#[derive(Debug)]
pub struct Param {
    pub name: String,
}

macro_rules! hashset {
    () => { 
        HashSet::new() 
    };
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert(String::from($val)); )*
         set
    }}
}

impl Pool {
    fn new(dims: Vec<usize>, size: usize) -> Pool {
        Pool((0..size)
             .map(|_| RefCell::new(Constant::empty(dims.clone())))
             .collect())
    }

    fn size(&self) -> usize {
        self.0.len()
    }

    fn get(&self, i: usize) -> RefMut<Constant> {
        match self.0.get(i) {
            Some(x) => x.borrow_mut(),
            None    => panic!("Can't get index {} from a pool with {} elements", 
                              i, self.size())
        }
    }
}


impl<'a> Index<usize> for Pool {
    type Output = RefCell<Constant>;

    fn index(&self, i: usize) -> &RefCell<Constant> { &self.0[i] }
}

impl Function {

    // Constructors

   pub fn new(value: Constant,
          params: HashSet<String>,
          body: Expr,
          pool_size: usize) -> Function {
       let dims = (&value).dims();
       Function {
           value: RefCell::new(value),
           params: params,
           body: Rc::new(body),
           pool: Pool::new(dims, pool_size)
       }
    }

    pub fn constant(value: Constant) -> Function {
        Function::new(value.clone(), hashset!(), Expr::Constant(value), 0)
    }

    //#[allow(dead_code)]
    //pub fn input(s: &str, dims: Vec<usize>) -> Function {
        //Function::new(None, hashset![s], 
                    //Expr::Input(Input {
                        //name: String::from(s),
                        //dims: dims,
                    //}))
    //}


    #[allow(dead_code)]
    pub fn param(s: &str, value: Constant) -> Function {
        Function::new(value, hashset![s], Expr::Param(Param { name: String::from(s) }), 0)
    }

    #[allow(dead_code)]
    pub fn scalar_param(s: &str, value: f32) -> Function {
        Function::param(s, Constant::Scalar(value)) 
    }

    #[allow(dead_code)]
    pub fn matrix_param(s: &str, height: usize, width: usize, vals: Vec<f32>) -> Function {
        Function::param(s, Constant::matrix(height, width, vals))
    }

    #[allow(dead_code)]
    pub fn single_val_param(s: &str, dims: Vec<usize>, val: f32) -> Function {
        Function::param(s, Constant::single_val(dims, val))
    }

    #[allow(dead_code)]
    pub fn random_param(s: &str, dims: Vec<usize>, lo: f32, hi: f32) -> Function {
        let value = Constant::random(dims, lo, hi);
        Function::param(s, value)
    }

    #[allow(dead_code)]
    pub fn random_matrix(dims: Vec<usize>, lo: f32, hi: f32) -> Function {
        Function::constant(Constant::random(dims, lo, hi))
    }

    #[allow(dead_code)]
    pub fn scalar(x: f32) -> Function {
        Function::constant(Constant::Scalar(x)) 
    }

    #[allow(dead_code)]
    pub fn matrix(height: usize, width: usize, values: Vec<f32>) -> Function {
        Function::constant(Constant::matrix(height, width, values))
    }

    #[allow(dead_code)]
    pub fn single_val_matrix(height: usize, width: usize, value: f32) -> Function {
        Function::constant(Constant::single_val(vec![height, width], value))
    }

    // accessors

    pub fn body(&self) -> &Expr {
        &*self.body
    }

    pub fn params(&self) -> &HashSet<String> {
        &self.params
    }

    pub fn value(&self) -> Ref<Constant> {
        self.value.borrow()
    }

    pub fn value_mut(&self) -> RefMut<Constant> {
        self.value.borrow_mut()
    }

    pub fn placeholder(&self, i: usize) -> RefMut<Constant> {
        self.pool.get(i)
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
