use constant::Constant;
use std::cell::{RefCell, Ref, RefMut};
use std::collections::HashSet;
use std::rc::Rc;
use std::ops::{Deref, DerefMut};
use std::ops::Index;

//#[derive(Debug, Clone)]
#[derive(Clone)]
struct Pool(Vec<RefCell<Constant>>);

//#[derive(Debug, Clone)]
#[derive(Clone)]
pub struct Function {
    value: RefCell<Constant>,
    params: HashSet<String>,
    body: Rc<Expr>,
    pool: Pool,
}

//#[derive(Debug)]
pub enum Expr {
    Constant(Constant),
    //Input(Input),
    Param(Param),
    //Neg(Function),
    //Sq(Function),
    //Abs(Function),
    //Signum(Function),
    //Sigmoid(Function),
    //Tanh(Function),
    //Add(Function, Function),
    Sub(Function, Function),
    Mul(Function, Function),
    //Dot(Function, Function, bool, bool),
}

//#[derive(Debug)]
//pub struct Input {
    //dims: Vec<u32>, 
    //pub name: String,
//}

#[derive(Debug)]
pub struct Param {
    pub name: String,
}

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert(String::from($val)); )*
         set
    }}
}

fn combine_params(f1: &Function, f2: &Function) -> HashSet<String> {
    let params1 = f1.params().clone();
    let params2 = f2.params().clone();
    return params1.union(&params2).cloned().collect()
}

impl Pool {
    fn new(dims: Vec<u32>, size: u32) -> Pool {
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

   fn new(value: Constant,
          params: HashSet<String>,
          body: Expr,
          pool_size: u32) -> Function {
       let dims = (&value).dims();
        Function {
            value: RefCell::new(value),
            params: params,
            body: Rc::new(body),
            pool: Pool::new(dims, pool_size)
        }
    }

    pub fn constant(value: Constant) -> Function {
        Function::new(value.clone(), HashSet::new(), Expr::Constant(value), 0)
    }

    pub fn sub(arg1: Function, arg2: Function) -> Function {
        let dummy = arg1.value().clone();
        Function::new(dummy.clone(),
                      combine_params(&arg1, &arg2),
                      Expr::Sub(arg1, arg2),
                      1) 
    }

    //#[allow(dead_code)]
    //pub fn input(s: &str, dims: Vec<u32>) -> Function {
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
    pub fn random_param(s: &str, dims: Vec<u32>, lo: f32, hi: f32) -> Function {
        let value = Constant::random(dims, lo, hi);
        Function::param(s, value)
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

    // accessors

    pub fn body(&self) -> &Expr {
        &*self.body
    }

    pub fn params(&self) -> &HashSet<String> {
        &self.params
    }

    //pub fn set_value(&self, value: Constant) {
        //self.value_mut().get_mut() = value;
    //}

    pub fn value(&self) -> Ref<Constant> {
        self.value.borrow()
    }

    pub fn value_mut(&self) -> RefMut<Constant> {
        self.value.borrow_mut()
    }

    //pub fn get_value(&self) -> Ref<Constant> {
        //self.value.borrow()
    //}

   //pub fn mutate_value(&self, f: &Fn(&Constant)) {
        //f(self.value_mut().deref_mut())
    //}

    //pub fn unwrap_value<'a>(&'a self) -> Constant {
        //self.value
    //}

    //pub fn alloc_placeholders(&self, c: Vec<Constant>) {
        //*self.placeholders.borrow_mut() = c;
    //}

    pub fn placeholder(&self, i: usize) -> RefMut<Constant> {
        self.pool.get(i)
    }

    //pub fn mutate_placeholder(&self, i: usize, f: &Fn(&mut Constant)) {
        //match self.placeholders.borrow_mut().get_mut(i) {
            //Some(ref mut placeholder) => f(placeholder),
            //None => panic!("Tried to mutate a placeholder that hasn't been assigned yet."),
        //}
    //}

    //pub fn num_placeholders(&self) -> usize {
        //self.placeholders.borrow().len()
    //}

    pub fn check_params(functions: Vec<&Function>) {
        for function in functions {
            match *function.body {
                Expr::Param(_) => {},
                _ => panic!("TODO: change this message")
                    //panic!("{} must be a param", &function),
            }
        }
    }
}
