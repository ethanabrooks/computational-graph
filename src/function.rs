use std::fmt;
use std::ops::{Neg, Add, Mul};
use constant::{Constant, copy_and_fill, new_constant, new_matrix};
use std::collections::{HashMap, HashSet};
use std::cell::{RefCell, Ref};
use std::rc::Rc;

type Shared<T> = Rc<RefCell<T>>;

mod shared {
    use std::{rc, cell};
    use function::Shared;

    pub fn new<T>(value: T) -> Shared<T> {
        rc::Rc::new(cell::RefCell::new(value))
    }
}

//// STRUCTS AND ENUMS

#[derive(Debug)]
struct Input {
    dims: Vec<i32>, 
    name: String,
}

#[derive(Debug)]
struct Param {
    value: Shared<Constant>,
    name: String,
}

#[derive(Debug)]
enum Expr {
    Constant(Constant),
    Input(Input),
    Param(Param),
    Neg(Function),
    Add(Function, Function),
    Mul(Function, Function),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub output: Shared<Option<Constant>>,
    pub params: HashSet<String>,
    body: Rc<Expr>,
}

//// TRAITS

fn write_with_parens(a: &Function, 
                     operator: &str,
                     b: &Function,  
                     f: &mut fmt::Formatter) -> fmt::Result {
    match *a.body.clone() {
        Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => match *b.body.clone() {
            Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                    write!(f, "{} {} {}", a, operator, b),
                _ => write!(f, "{} {} ({})", a, operator, b),
        },
        _  => match *b.body.clone() {
                Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                    write!(f, "({}) {} {}", a, operator, b),
                _ => write!(f, "({}) {} ({})", a, operator, b),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Expr::Constant(ref c) => write!(f, "{}", c), 
            Expr::Input(ref i) => write!(f, "{}", i.name),
            Expr::Param(ref p) => write!(f, "{}:={}", p.name, 
                                         get_shared(&p.value).clone()),
            Expr::Neg(ref x) => match *x.body.clone() {
                Expr::Constant(_) | Expr::Input(_)  => write!(f, "-{}", x),
                _  => write!(f, "-({})", x),
            },
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            Expr::Mul(ref a, ref b) => write_with_parens(a, "*", b, f),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", *self.body.clone())
    }
}


fn un_apply(expr: &Fn(Function) -> Expr, f: &Function) -> Function {
    Function {
        output: shared::new(None),
        params: f.params.clone(),
        body: Rc::new(expr(f.clone())),
    }
}

fn bin_apply(expr: &Fn(Function, Function) -> Expr, 
             f1: &Function, f2: &Function) -> Function {
    let params1 = f1.params.clone();
    let params2 = f2.params.clone();

    Function {
        output: shared::new(None),
        params: params1.union(&params2).cloned().collect(), // TODO: does this generalize?
        body: Rc::new(expr(f1.clone(), f2.clone())),
    }
}

impl Neg for Function {
    type Output = Function;
    fn neg(self) -> Function { -&self }
}

impl Add for Function {
    type Output = Function;
    fn add(self, other: Function) -> Function { &self + &other }
}

impl Mul for Function {
    type Output = Function;
    fn mul(self, other: Function) -> Function { &self * &other }
}

impl <'a> Neg for &'a Function {
    type Output = Function;
    fn neg(self) -> Function {
        un_apply(&|f| Expr::Neg(f), self)
    }
}

impl<'a> Add for &'a Function {
    type Output = Function;
    fn add(self, other: &Function) -> Function {
        bin_apply(&|f1, f2| Expr::Add(f1, f2), self, other) 
    }
}

impl<'a> Mul for &'a Function {
    type Output = Function;
    fn mul(self, other: &Function) -> Function {
        bin_apply(&|f1, f2| Expr::Mul(f1, f2), self, other) 
    }
}

//// constructors

pub fn input(s: &str, dims: Vec<i32>) -> Function {
    let mut params = HashSet::new();
    params.insert(String::from(s));
    Function {
        output: shared::new(None),
        params: params,
        body: Rc::new(Expr::Input(Input {
                name: String::from(s),
                dims: dims,
        }))
    }
}

pub fn param(s: &str, value: Constant) -> Function {
    let mut params = HashSet::new();
    params.insert(String::from(s));
    Function {
        output: shared::new(Some(value.clone())),
        params: params,
        body: Rc::new(Expr::Param(Param {
                name: String::from(s),
                value: shared::new(value.clone()),
        }))
    }
}

pub fn scalar(x: f32) -> Function {
    Function {
        output: shared::new(Some(Constant::Scalar(x))),
        params: HashSet::new(),
        body: Rc::new(Expr::Constant(Constant::Scalar(x))), 
    }
}

pub fn matrix(height: i32, width: i32, values: Vec<f32>) -> Function {
    let m = new_matrix(height, width, values);
    Function {
        output: shared::new(Some(m.clone())),
        params: HashSet::new(),
        body: Rc::new(Expr::Constant(m.clone())),
    }
}

//// MAIN FUNCTIONS

fn get_shared<T>(s: &Shared<T>) -> Ref<T> { s.borrow() }


fn get_output(f: &Function) -> Constant {
    match *get_shared(&f.output) {
        Some(ref x) => x.clone(), // Can we avoid this clone?
        None => panic!("Need to run `assign_outputs` before `grad`"),
    }
}

// TODO: return an expr instead of a Constant
pub fn grad(f: &Function, param: &str) -> Constant {
    match f.params.contains::<str>(&param) {
        false => Constant::Scalar(0.),
        true => match *f.body { 
            Expr::Constant(ref c) => copy_and_fill(c, 0.), 
            Expr::Input(ref i) => new_constant(&i.dims, 0.),
            Expr::Param(ref p) => {
                copy_and_fill(&p.value.borrow(),
                              match param == p.name {
                                  true => 1., 
                                  false => 0.,
                              })
            },
            Expr::Neg(ref f) => -grad(&f, param),
            Expr::Add(ref f1, ref f2) => grad(&f1, param) + grad(&f2, param),
            Expr::Mul(ref f1, ref f2) => grad(&f1, param) * get_output(&f2) +
                                         grad(&f2, param) * get_output(&f1),
        }
    }
}

fn apply_to_branches(f: &Fn(Constant, Constant) -> Constant, 
                     args: &HashMap<&str, Constant>,
                     f1: &Function, 
                     f2: &Function) -> Option<Constant> {
    match (eval(&f1, args), eval(&f2, args)) {
        (Some(x1), Some(x2)) => Some(f(x1, x2)),
        _ => None,
    }
}


pub fn eval(f: &Function, args: &HashMap<&str, Constant>) -> Option<Constant> {
    match *f.body { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Input(ref i) => args.get::<str>(&i.name).map(|x| x.clone()),
        Expr::Param(ref p) => Some(get_shared(&p.value).clone()),
        Expr::Neg(ref f) => eval(&f, args).map(|x| -x),
        Expr::Add(ref f1, ref f2) => apply_to_branches(&|x, y| x + y, args, f1, f2),
        Expr::Mul(ref f1, ref f2) => apply_to_branches(&|x, y| x * y, args, f1, f2),
    }
}


fn assign_and_apply(f: &Fn(Constant, Constant) -> Constant, 
                    args: &HashMap<&str, Constant>,
                    f1: &Function, f2: &Function) -> Option<Constant> {
    assign_outputs(f1, args);
    assign_outputs(f2, args);
    apply_to_branches(f, args, f1, f2)
}


pub fn assign_outputs(f: &Function, args: &HashMap<&str, Constant>) {
    *f.output.borrow_mut() = match *f.body.clone() { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Input(ref arg) => args.get::<str>(&arg.name).map(|x| x.clone()),
        Expr::Param(ref p) => Some(get_shared(&p.value).clone()),
        Expr::Neg(ref f1) => {
            assign_outputs(f1, args);
            Some(-get_output(f1))
        }
        Expr::Add(ref f1, ref f2) => assign_and_apply(&|x, y| x + y, args, f1, f2),
        Expr::Mul(ref f1, ref f2) => assign_and_apply(&|x, y| x * y, args, f1, f2),
    }
}

pub fn minimize(f: &Function, learn_rate: f32, iters: i32) {
    for _ in 0..iters {
        backprop(f, &get_output(f), learn_rate);
    }
}

pub fn maximize(f: &Function, learn_rate: f32, iters: i32) {
    minimize(&-f, learn_rate, iters);
}

fn backprop(f: &Function, error: &Constant, learn_rate: f32) {
    if f.params.is_empty() { return; }
    match *f.body.clone() {
        Expr::Param(ref p) => { 
            let mut value = p.value.borrow_mut();
            *value -= &Constant::Scalar(learn_rate) * error; 
        }
        Expr::Neg(ref f1) => backprop(f1, &-error, learn_rate),
        Expr::Add(ref f1, ref f2) => {
            backprop(f1, error, learn_rate);
            backprop(f2, error, learn_rate);
        }
        Expr::Mul(ref f1, ref f2) => {
            backprop(f1, &(&get_output(f2) * error), learn_rate);
            backprop(f2, &(&get_output(f1) * error), learn_rate);
        }
        _ => return,
    }
}
