use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::copy_and_fill;
use constant::new_constant;
use std::collections::HashMap;
use std::collections::HashSet;

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
    //sub(Expr, Expr),
}

#[derive(Debug)]
pub struct Function<'a> {
    output: Option<Constant>,
    pub variables: HashSet<String>,
    body: Expr<'a>,
}

impl<'a> Neg for &'a Function<'a> {
    type Output = Function<'a>;
    fn neg(self) -> Function<'a> {
        Function {
            output: None,
            variables: self.variables.clone(),
            body: Expr::Neg(self),
        }
    }
}

impl<'a> Add for &'a Function<'a>{
    type Output = Function<'a>;
    fn add(self, other: &'a Function<'a>) -> Function<'a> {
        let vars1 = self.variables.clone();
        let vars2 = other.variables.clone();

        Function {
            output: None,
            variables: vars1.union(&vars2).cloned().collect(),
            body: Expr::Add(self, other),
        }
    }
}

pub fn variable<'a>(s: &str, dims: Vec<i32>) -> Function<'a> {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Function {
        output: None,
        variables: vars,
        body: Expr::Variable(Variable {
                name: String::from(s),
                dims: dims,
        })
    }

}

pub fn scalar<'a>(x: f32) -> Function<'a> {
    Function {
        output: Some(Constant::Scalar(x)),
        variables: HashSet::new(),
        body: Expr::Constant(Constant::Scalar(x)), 
    }
}


pub fn grad<'a>(f: &Function<'a>, var: &str) -> Constant {
    match f.variables.contains::<str>(&var) {
        false => Constant::Scalar(0.),
        true => match f.body { 
            Expr::Constant(ref c) => copy_and_fill(c, 0.), 
            Expr::Variable(ref v) => new_constant(v.dims.clone(), 1.),
            Expr::Neg(ref f) => -grad(&f, var),
            Expr::Add(ref f1, ref f2) => grad(&f1, var) + grad(&f2, var),
        }
    }
}

pub fn eval<'a>(f: &Function<'a>, args: &HashMap<&str, Constant>) -> Option<Constant> {
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

