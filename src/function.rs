use std::fmt;
use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::Constant::matrix;
use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::PartialEq;

#[derive(Debug)]
pub struct Variable {
    pub gradient: Option<Constant>,
    pub name: String,
}

#[derive(Debug)]
enum Expr {
    constant(Constant),
    variable(Variable),
    neg(Box<Function>),
    add(Box<Function>, Box<Function>),
    //sub(Expr, Expr),
}

#[derive(Debug)]
pub struct Function {
	output: Option<Constant>,
	variables: HashSet<String>,
	body: Expr,
}

impl Neg for Box<Function> {
    type Output = Box<Function>;
    fn neg(self) -> Box<Function> {
        Box::new(Function {
            output: None,
            variables: self.variables.clone(),
            body: Expr::neg(self),
        })
    }
}

impl Add for Box<Function> {
    type Output = Box<Function>;
    fn add(self, other: Box<Function>) -> Box<Function> {
        let mut vars = self.variables.clone();
        vars.union(&mut other.variables.clone());

        Box::new(Function {
            output: None,
            variables: vars,
            body: Expr::add(self, other),
        })
    }
}

pub fn variable(s: &str) -> Box<Function> {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Box::new(Function {
        output: None,
        variables: vars,
        body: Expr::variable(Variable {
                name: String::from(s),
                gradient: None,
        })
    })

}

pub fn scalar(x: f32) -> Box<Function> {
    Box::new(Function {
        output: Some(Constant::scalar(x)),
        variables: HashSet::new(),
        body: Expr::constant(Constant::scalar(x)), 
    })

}


pub fn grad(f: &Box<Function>, var: &Variable) -> Constant {
    match f.variables.contains(&var.name) {
        false => Constant::scalar(0.),
        true => match f.body { 
            Expr::constant(_) => Constant::scalar(0.), //TODO: accomodate matrices
            Expr::variable(ref var) => Constant::scalar(1.),
            Expr::neg(ref f) => -grad(&f, var),
            Expr::add(ref f1, ref f2) => grad(&f1, var) + grad(&f2, var),
        }
    }
}

pub fn eval(f: &Box<Function>, args: &HashMap<String, Constant>) -> Option<Constant> {
    match f.body { 
        Expr::constant(ref x) => Some(x.clone()),
        Expr::variable(ref var) => args.get(&var.name).map(|x| x.clone()),
        Expr::neg(ref f) => eval(&f, args).map(|x| -x),
        Expr::add(ref f1, ref f2) => 
            match (eval(&f1, args), eval(&f2, args)) {
                (Some(x1), Some(x2)) => Some(x1 + x2),
                _ => None,
            }
    }
}

