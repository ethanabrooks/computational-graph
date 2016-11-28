use std::fmt;
use std::ops::{Neg, Add, Mul};
use constant::{Constant, copy_and_fill, new_constant, new_matrix};
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;

#[derive(Debug)]
struct Variable {
    dims: Vec<i32>, 
    name: String,
}

#[derive(Debug)]
enum Expr<'a> {
    Constant(Constant),
    Variable(Variable),
    Neg(&'a Function<'a>),
    Add(&'a Function<'a>, &'a Function<'a>),
    Mul(&'a Function<'a>, &'a Function<'a>),
}

#[derive(Debug)]
pub struct Function<'a> {
    output: RefCell<Option<Constant>>,
    pub variables: HashSet<String>,
    body: Expr<'a>,
}

impl<'a> fmt::Display for Function<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.body)
    }
}

fn write_with_parens<'a>(a: &Function<'a>, 
                     operator: &str,
                     b: &Function<'a>,  
                     f: &mut fmt::Formatter) -> fmt::Result {
    match a.body {
        Expr::Constant(_) | Expr::Variable(_) => match b.body {
            Expr::Constant(_) | Expr::Variable(_) => 
                    write!(f, "{} {} {}", a, operator, b),
                _ => write!(f, "{} {} ({})", a, operator, b),
        },
        _  => match b.body {
                Expr::Constant(_) | Expr::Variable(_) => 
                    write!(f, "({}) {} {}", a, operator, b),
                _ => write!(f, "({}) {} ({})", a, operator, b),
        }
    }
}

impl<'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Expr::Constant(ref c) => write!(f, "{}", c), 
            Expr::Variable(ref v) => write!(f, "{}", v),
            Expr::Neg(ref x) => match x.body {
                Expr::Constant(_) | Expr::Variable(_)  => write!(f, "-{}", x),
                _  => write!(f, "-({})", x),
            },
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            Expr::Mul(ref a, ref b) => write_with_parens(a, "*", b, f),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<'a> Neg for &'a Function<'a> {
    type Output = Function<'a>;
    fn neg(self) -> Function<'a> {
        Function {
            output: RefCell::new(None),
            variables: self.variables.clone(),
            body: Expr::Neg(self),
        }
    }
}

impl<'a> Add for &'a Function<'a> {
    type Output = Function<'a>;
    fn add(self, other: &'a Function<'a>) -> Function<'a> {
        let vars1 = self.variables.clone();
        let vars2 = other.variables.clone();

        Function {
            output: RefCell::new(None),
            variables: vars1.union(&vars2).cloned().collect(),
            body: Expr::Add(self, other),
        }
    }
}

impl<'a> Mul for &'a Function<'a> {
    type Output = Function<'a>;
    fn mul(self, other: &'a Function<'a>) -> Function<'a> {
        let vars1 = self.variables.clone();
        let vars2 = other.variables.clone();

        Function {
            output: RefCell::new(None),
            variables: vars1.union(&vars2).cloned().collect(),
            body: Expr::Mul(self, other),
        }
    }
}

pub fn variable<'a>(s: &str, dims: Vec<i32>) -> Function<'a> {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Function {
        output: RefCell::new(None),
        variables: vars,
        body: Expr::Variable(Variable {
                name: String::from(s),
                dims: dims,
        })
    }
}

pub fn scalar<'a>(x: f32) -> Function<'a> {
    Function {
        output: RefCell::new(Some(Constant::Scalar(x))),
        variables: HashSet::new(),
        body: Expr::Constant(Constant::Scalar(x)), 
    }
}

pub fn matrix<'a>(height: i32, width: i32, values: Vec<f32>) -> Function<'a> {
    let m = new_matrix(height, width, values);
    Function {
        output: RefCell::new(Some(m.clone())),
        variables: HashSet::new(),
        body: Expr::Constant(m.clone()),
    }
}

fn get_output(f: &Function) -> Constant {
    // Can we avoid this clone?
    f.output.clone().into_inner().expect("Need to run `assign_outputs` before `grad`")
}

pub fn grad<'a>(f: &Function<'a>, var: &str) -> Constant {
    match f.variables.contains::<str>(&var) {
        false => Constant::Scalar(0.),
        true => match f.body { 
            Expr::Constant(ref c) => copy_and_fill(c, 0.), 
            Expr::Variable(ref v) => new_constant(&v.dims, 1.),
            Expr::Neg(ref f) => -grad(&f, var),
            Expr::Add(ref f1, ref f2) => grad(&f1, var) + grad(&f2, var),
            Expr::Mul(ref f1, ref f2) => grad(&f1, var) * get_output(&f2) +
                                         grad(&f2, var) * get_output(&f1),
        }
    }
}

fn apply_to_branches<'a>(f: &Fn(Constant, Constant) -> Constant, 
                     args: &HashMap<&str, Constant>,
                     f1: &Function<'a>, 
                     f2: &Function<'a>) -> Option<Constant> {
    match (eval(&f1, args), eval(&f2, args)) {
        (Some(x1), Some(x2)) => Some(f(x1, x2)),
        _ => None,
    }
}


pub fn eval<'a>(f: &Function<'a>, args: &HashMap<&str, Constant>) -> Option<Constant> {
    match f.body { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Variable(ref var) => args.get::<str>(&var.name).map(|x| x.clone()),
        Expr::Neg(ref f) => eval(&f, args).map(|x| -x),
        Expr::Add(ref f1, ref f2) => apply_to_branches(&|x, y| x + y, args, f1, f2),
        Expr::Mul(ref f1, ref f2) => apply_to_branches(&|x, y| x * y, args, f1, f2),
    }
}


fn assign_and_apply<'a>(f: &Fn(Constant, Constant) -> Constant, 
                                        args: &HashMap<&str, Constant>,
                                        f1: &Function<'a>, 
                                        f2: &Function<'a>) -> Option<Constant> {
    assign_outputs(f1, args);
    assign_outputs(f2, args);
    apply_to_branches(f, args, f1, f2)
}


pub fn assign_outputs<'a>(f: &Function<'a>, args: &HashMap<&str, Constant>) {
    *f.output.borrow_mut() = match f.body { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Variable(ref var) => args.get::<str>(&var.name).map(|x| x.clone()),
        Expr::Neg(ref f1) => {
            assign_outputs(f1, args);
            f1.output.borrow().clone().map(|x| -x)
        }
        Expr::Add(ref f1, ref f2) => assign_and_apply(&|x, y| x + y, args, f1, f2),
        Expr::Mul(ref f1, ref f2) => assign_and_apply(&|x, y| x * y, args, f1, f2),
    }
}
