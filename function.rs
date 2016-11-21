use constant::Constant;
use constant::Constant::scalar;
use constant::Constant::matrix;
use std::collections::HashMap;
use std::cmp::PartialEq;

struct Variable {
    name: String,
    gradient: Constant,
}

impl PartialEq for Variable {
    fn eq(&self, other: &Variable) -> bool { 
        self.name == other.name
    }
}

enum Expr {
    constant(Constant),
    variable(Variable),
    neg(Box<Function>),
    add(Box<Function>, Box<Function>),
    //sub(Expr, Expr),
}

struct Function {
	variables: Vec<Variable>,
	body: Expr,
	output: Option<Constant>,
}

fn grad(f: Function, var: &Variable) -> Constant {
    match f.variables.contains(&var) {
        false => scalar(0.),
        true => match f.body { 
            Expr::constant(_) => scalar(0.),
            Expr::variable(var) => scalar(1.),
            Expr::neg(f) => -grad(*f, var),
            Expr::add(f1, f2) => grad(*f1, var) + grad(*f2, var),
        }
    }
}

fn eval(f: Function, args: HashMap<String, Constant>) -> Option<Constant> {
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

