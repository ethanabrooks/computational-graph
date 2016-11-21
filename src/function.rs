use std::fmt;
use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::Constant::matrix;
use std::collections::HashMap;
use std::cmp::PartialEq;

#[derive(Debug)]
pub struct Variable {
    pub gradient: Option<Constant>,
    pub name: String,
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

//impl fmt::Display for Box<Function> {
	//fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		//write!(f, "{:#?}", self.body)
	//}
//}


//impl fmt::Display for Expr {
	//fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		//match *self {
			//Expr::constant(x) => write!(f, "{:?}", x),
			//Expr::variable(v) => write!(f, "{}", v.name),
			//Expr::neg(f) => write!(f, "-{:?}", f),
			//Expr::add(f1, f2) =>  write!(f, "{:?} + {:?}", f1, f2),
		//}
	//}
//}




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
        vars.append(&mut other.variables.clone());

        Box::new(Function {
            output: None,
            variables: vars,
            body: Expr::add(self, other),
        })
    }
}

pub fn variable(n: &str) -> Box<Function> {
    Box::new(Function {
        output: None,
        variables: vec![String::from(n)],
        body: Expr::variable(Variable {
                name: String::from(n),
                gradient: None,
        })
    })

}

pub fn scalar(x: f32) -> Box<Function> {
    Box::new(Function {
        output: Some(Constant::scalar(x)),
        variables: vec![],
        body: Expr::constant(Constant::scalar(x)), 
    })

}


pub fn grad(f: Function, var: &Variable) -> Constant {
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

pub fn eval(f: Function, args: &HashMap<String, Constant>) -> Option<Constant> {
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

