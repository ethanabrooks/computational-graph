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
enum Expr<'a> {
    Constant(Constant),
    Variable(Variable),
    Neg(&'a Function),
    Add(&'a Function, &'a Function),
    //sub(Expr, Expr),
}

#[derive(Debug)]
pub struct Function<'a> {
	output: Option<Constant>,
	variables: HashSet<String>,
	body: Expr<'a>,
}

impl Neg for &'a Function {
    type Output = &'a Function;
    fn neg(self) -> &'a Function {
        Function {
            output: None,
            variables: self.variables.clone(),
            body: Expr::Neg(self),
        }
    }
}

impl Add for &'a Function {
    type Output = &'a Function;
    fn add(self, other: &'a Function) -> &'a Function {
        let mut vars = self.variables.clone();
        vars.union(&mut other.variables.clone());

        Function {
            output: None,
            variables: vars,
            body: Expr::Add(self, other),
        }
    }
}

pub fn variable<'a>(s: &str) -> &'a Function> {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Function {
        output: None,
        variables: vars,
        body: Expr::Variable<'a>(Variable {
                name: String::from(s),
                gradient: None,
        }
    })

}

pub fn scalar(x: f32) -> &'a Function {
    Function {
        output: Some(Constant::Scalar(x)),
        variables: HashSet::new(),
        body: Expr::Constant(Constant::Scalar(x)), 
    }
}


pub fn grad<'a>(f: &'a Function>, var: &str) -> Constant {
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

