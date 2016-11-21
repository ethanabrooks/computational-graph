use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::Constant::matrix;
use std::collections::HashMap;
use std::cmp::PartialEq;

#[derive(Debug)]
pub struct Variable {
    gradient: Option<Constant>,
    name: String,
}

impl PartialEq for Variable {
    fn eq(&self, other: &Variable) -> bool { 
        self.name == other.name
    }
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
	variables: Vec<String>,
	body: Expr,
}


impl Neg for Function {
    type Output = Function;
    fn neg(self) -> Function {
        Function {
            output: None,
            variables: vec![], //TODO!!
            body: Expr::neg(Box::new(self)),
        }
    }
}

impl Add for Function {
    type Output = Function;
    fn add(self, other: Function) -> Function {
        Function {
            output: None,
            variables: vec![], //TODO!!
            body: Expr::add(Box::new(self), Box::new(other)),
        }
    }
}

pub fn variable(n: &str) -> Function {
    Function {
        output: None,
        variables: vec![String::from(n)],
        body: Expr::variable(Variable {
                name: String::from(n),
                gradient: None,
        })
    }

}

pub fn scalar(x: f32) -> Function {
    Function {
        output: Some(Constant::scalar(x)),
        variables: vec![],
        body: Expr::constant(Constant::scalar(x)), 
    }

}


fn grad(f: Function, var: &Variable) -> Constant {
    match f.variables.contains(&var.name) {
        false => Constant::scalar(0.),
        true => match f.body { 
            Expr::constant(_) => Constant::scalar(0.),
            Expr::variable(var) => Constant::scalar(1.),
            Expr::neg(f) => -grad(*f, var),
            Expr::add(f1, f2) => grad(*f1, var) + grad(*f2, var),
        }
    }
}

fn eval(f: Function, args: &HashMap<String, Constant>) -> Option<Constant> {
    match f.body { 
        Expr::constant(x) => Some(x),
        Expr::variable(var) => args.get(&var.name).map(|x| x.clone()),
        Expr::neg(f) => eval(*f, args).map(|x| -x),
        Expr::add(f1, f2) => 
            match (eval(*f1, args), eval(*f2, args)) {
                (Some(x1), Some(x2)) => Some(x1 + x2),
                _ => None,
            }
    }
}

