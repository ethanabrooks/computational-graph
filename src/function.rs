use std::fmt;
use std::ops::{Neg, Add, Mul};
use constant::{Constant, copy_and_fill, new_matrix, all_equal};
use std::collections::{HashMap, HashSet};
use std::cell::{RefCell, Ref};
use std::rc::Rc;
use std::io;
use std::io::Write;

//// MACROS

macro_rules! hashset {
    ($( $val: expr ),*) => {{
         let mut set = HashSet::new();
         $( set.insert($val); )*
         set
    }}
}

//// STRUCTS AND ENUMS

type Shared<T> = Rc<RefCell<T>>;

mod shared {
    use std::{rc, cell};
    use function::Shared;

    pub fn new<T>(value: T) -> Shared<T> {
        rc::Rc::new(cell::RefCell::new(value))
    }
}


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
    Abs(Function),
    Sign(Function),
    Add(Function, Function),
    Mul(Function, Function),
}

#[derive(Debug, Clone)]
pub struct Function {
    output: Shared<Option<Constant>>,
    params: HashSet<String>,
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
            Expr::Param(ref p) => write!(f, "{}≔{}", p.name, 
                                         get_shared(&p.value).clone()),
            Expr::Neg(ref x) => match *x.body.clone() {
                Expr::Constant(_) | Expr::Input(_)  => write!(f, "-{}", x),
                _  => write!(f, "-({})", x),
            },
            Expr::Abs(ref x) => write!(f, "|{}|", x),
            Expr::Sign(ref x) => write!(f, "sign({})", x),
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            Expr::Mul(ref a, ref b) => write_with_parens(a, "×", b, f),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", *self.body.clone())
    }
}


impl Function {
    fn apply(&self, expr: &Fn(Function) -> Expr) -> Function {
        Function {
            output: shared::new(None),
            params: self.params.clone(),
            body: Rc::new(expr(self.clone())),
        }
    }

    fn all_equal(&self, val:f32) -> bool {
        match *self.body {
            Expr::Constant(ref c) => all_equal(c, val),
            _                     => false
        }
    }

    pub fn abs(&self) -> Function {
        self.apply(&|f| Expr::Abs(f))
    }

    fn signum(&self) -> Function {
        self.apply(&|f| Expr::Sign(f))
    }
}

fn bin_apply(expr: &Fn(Function, Function) -> Expr, 
             f1: &Function, f2: &Function, 
             identity: f32) -> Function {
    match (&*f1.body, &*f2.body) {

        // optimization to combine constants
        (&Expr::Constant(_), &Expr::Constant(_)) => {
            new_constant(new_function(None, HashSet::new(), expr(f1.clone(), f2.clone()))
                         .eval(&HashMap::new()).unwrap())
        }, 
        _ => {

            // optimization to eliminate identities
            if f1.all_equal(identity) {
                f2.clone()
            } else if f2.all_equal(identity) {
                f1.clone()
            } else {
                let params1 = f1.params.clone();
                let params2 = f2.params.clone();

                new_function(None, params1.union(&params2).cloned().collect(),
                            expr(f1.clone(), f2.clone()))
            }
        }
    }
}

// TODO: macros!
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

        // optimization to eliminate -0
        if self.all_equal(0.) {
            self.clone()
        } else {
            self.apply(&|f| Expr::Neg(f))
        }
    }
}

impl<'a> Add for &'a Function {
    type Output = Function;
    fn add(self, other: &Function) -> Function {
        bin_apply(&|f1, f2| Expr::Add(f1, f2), self, other, 0.) 
    }
}

impl<'a> Mul for &'a Function {
    type Output = Function;
    fn mul(self, other: &Function) -> Function {

        // optimization to eliminate multiplication by 0
        if self.all_equal(0.) {
            return self.clone()
        } 
        if other.all_equal(0.) {
            return other.clone()
        }
        bin_apply(&|f1, f2| Expr::Mul(f1, f2), self, other, 1.) 
    }
}

//// constructors

fn new_function(output: Option<Constant>, params: HashSet<String>, body: Expr) 
-> Function {
    Function {
        output: shared::new(output),
        params: params,
        body: Rc::new(body),
    }
}

fn new_constant(value: Constant) -> Function {
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
    new_function(None, params, 
                 Expr::Param(Param {
                     name: String::from(s),
                     value: shared::new(value),
                 }))
}

#[allow(dead_code)]
pub fn scalar(x: f32) -> Function {
    new_constant(Constant::Scalar(x)) 
}

#[allow(dead_code)]
pub fn matrix(height: i32, width: i32, values: Vec<f32>) -> Function {
    new_constant(new_matrix(height, width, values))
}

//// MAIN FUNCTIONS

fn get_shared<T>(s: &Shared<T>) -> Ref<T> { s.borrow() }

fn apply_to_branches(f: &Fn(Constant, Constant) -> Constant, 
                    args: &HashMap<&str, Constant>,
                    f1: &Function, 
                    f2: &Function) -> Option<Constant> {
    match (f1.eval(args), f2.eval(args)) {
        (Some(x1), Some(x2)) => Some(f(x1, x2)),
        _ => None,
    }
}

fn assign_and_apply(f: &Fn(Constant, Constant) -> Constant, 
                    args: &HashMap<&str, Constant>,
                    f1: &Function, f2: &Function) -> Option<Constant> {
    f1.assign_outputs(args);
    f2.assign_outputs(args);
    apply_to_branches(f, args, f1, f2)
}

impl Function {
    pub fn get_output(&self) -> Constant {
        match *get_shared(&self.output) {
            Some(ref x) => x.clone(),
            None => panic!("Need to run `assign_outputs` before `grad`"),
        }
    }

    pub fn grad(&self, param: &str) -> Function {
        if self.params.contains::<str>(&param) {
            match *self.body { 
                Expr::Neg(ref f) => -f.grad(param),
                Expr::Abs(ref f) => 
                    f.signum() * f.grad(param),
                Expr::Sign(ref f) => panic!("signum is nondifferentiable"),
                Expr::Add(ref f1, ref f2) => f1.grad(param) + f2.grad(param),
                Expr::Mul(ref f1, ref f2) => &f1.grad(param) * f2 +
                                             &f2.grad(param) * f1,
                Expr::Param(ref p) => new_constant(
                    copy_and_fill(&*get_shared(&p.value), 1.)
                    ),
                Expr::Constant(_)| Expr::Input(_) => panic!("should never reach here"),
            }
        } else {
            scalar(0.)
        }
    }

    pub fn eval(&self, args: &HashMap<&str, Constant>) -> Option<Constant> {
        match *self.body { 
            Expr::Constant(ref x) => Some(x.clone()),
            Expr::Input(ref i) => args.get::<str>(&i.name).map(|x| x.clone()),
            Expr::Param(ref p) => Some(get_shared(&p.value).clone()),
            Expr::Neg(ref f) => f.eval(args).map(|x| -x),
            Expr::Abs(ref f) => f.eval(args).map(|x| x.abs()),
            Expr::Sign(ref f) => f.eval(args).map(|x| x.signum()),
            Expr::Add(ref f1, ref f2) => apply_to_branches(&|x, y| x + y, args, f1, f2),
            Expr::Mul(ref f1, ref f2) => apply_to_branches(&|x, y| x * y, args, f1, f2),
        }
    }


    pub fn assign_outputs(&self, args: &HashMap<&str, Constant>) {
        *self.output.borrow_mut() = match *self.body { 
            Expr::Constant(ref x) => Some(x.clone()),
            Expr::Input(ref arg) => args.get::<str>(&arg.name).cloned(),
            Expr::Param(ref p) => Some(get_shared(&p.value).clone()),
            Expr::Neg(ref f1) => {
                f1.assign_outputs(args);
                Some(-f1.get_output().clone())
            }
            Expr::Abs(ref f1) => {
                f1.assign_outputs(args);
                Some(f1.get_output().abs().clone())
            }
            Expr::Sign(ref f1) => {
                writeln!(&mut io::stderr(), "WARN: Sign is non-differentiable.
                Running `backprop` on this function will cause an error");
                f1.assign_outputs(args);
                Some(f1.get_output().signum().clone())
            }
            Expr::Add(ref f1, ref f2) => assign_and_apply(&|x, y| x + y, args, f1, f2),
            Expr::Mul(ref f1, ref f2) => assign_and_apply(&|x, y| x * y, args, f1, f2),
        }
    }

    #[allow(dead_code)]
    pub fn minimize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        for _ in 0..iters {
            self.assign_outputs(args);
            self.backprop(&self.get_output(), learn_rate);
            println!("{}", self.get_output());
        }
    }

    #[allow(dead_code)]
    pub fn maximize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        self.abs().minimize(args, learn_rate, iters);
    }

    fn backprop(&self, error: &Constant, learn_rate: f32) {
        if self.params.is_empty() { return; }
        match *self.body.clone() {
            Expr::Param(ref p) => { 
                let mut value = p.value.borrow_mut();
                //println!("value: {}", *value);
                //println!("error: {}", error);
                *value -= &Constant::Scalar(learn_rate) * error; 
            }
            Expr::Neg(ref f1) => f1.backprop(&-error, learn_rate),
            Expr::Abs(ref f1) => f1.backprop(&error.abs(), learn_rate),
            Expr::Sign(ref f1) => panic!("sign is not differentiable"),
            Expr::Add(ref f1, ref f2) => {
                f1.backprop(error, learn_rate);
                f2.backprop(error, learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.backprop(&(&f2.get_output() * error), learn_rate);
                f2.backprop(&(&f1.get_output() * error), learn_rate);
            }
            Expr::Constant(_)| Expr::Input(_) => return,
        }
    }
}
