use std::fmt;
use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::Constant::Matrix;
use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::PartialEq;

#[derive(Debug)]
struct Variable {
    gradient: Option<Constant>,
    name: String,
}

#[derive(Debug)]
enum Expr {
    Constant(Constant),
    Variable(Variable),
    Neg(Box<Function>),
    Add(Box<Function>, Box<Function>),
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
            body: Expr::Neg(self),
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
            body: Expr::Add(self, other),
        })
    }
}

pub fn variable(s: &str) -> Box<Function> {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Box::new(Function {
        output: None,
        variables: vars,
        body: Expr::Variable(Variable {
                name: String::from(s),
                gradient: None,
        })
    })

}

pub fn scalar(x: f32) -> Box<Function> {
    Box::new(Function {
        output: Some(Constant::Scalar(x)),
        variables: HashSet::new(),
        body: Expr::Constant(Constant::Scalar(x)), 
    })
}


pub fn grad(f: &Box<Function>, var: &str) -> Constant {
    match f.variables.contains::<str>(&var) {
        false => Constant::Scalar(0.),
        true => match f.body { 
            Expr::Constant(_) => Constant::Scalar(0.), //TODO: accomodate matrices
            Expr::Variable(_) => Constant::Scalar(1.),  //TODO: accomodate matrices
            Expr::Neg(ref f) => -grad(&f, var),
            Expr::Add(ref f1, ref f2) => grad(&f1, var) + grad(&f2, var),
        }
    }
}

pub fn eval(f: &Box<Function>, args: &HashMap<&str, Constant>) -> Option<Constant> {
    match f.body { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Variable(ref var) => args.get::<str>(&var.name).map(|x| x.clone()),
        Expr::Neg(ref f) => eval(&f, args).map(|x| -x),
        Expr::Add(ref f1, ref f2) => 
            match (eval(&f1, args), eval(&f2, args)) {
                (Some(x1), Some(x2)) => Some(x1 + x2),
                _ => None,
            }
    }
}

