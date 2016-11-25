use std::fmt;
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
enum Expr {
    Constant(Constant),
    Variable(Variable),
    Neg(Function),
    Add(Function, Function),
    //Mul(Function, Function),
}

#[derive(Debug)]
pub struct Function {
    output: Option<Constant>,
    pub variables: HashSet<String>,
    body: Box<Expr>,
}

impl<'a> fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.body)
    }
}

fn write_with_parens(a: &Function, 
                     operator: &str,
                     b: &Function,  
                     f: &mut fmt::Formatter) -> fmt::Result {
    match *a.body {
        Expr::Constant(_) | Expr::Variable(_) => match *b.body {
            Expr::Constant(_) | Expr::Variable(_) => 
                    write!(f, "{} {} {}", a, operator, b),
                _ => write!(f, "{} {} ({})", a, operator, b),
        },
        _  => match *b.body {
                Expr::Constant(_) | Expr::Variable(_) => 
                    write!(f, "({}) {} {}", a, operator, b),
                _ => write!(f, "({}) {} ({})", a, operator, b),
        }
    }
}

impl<'a> fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Expr::Constant(ref c) => write!(f, "{}", c), 
            Expr::Variable(ref v) => write!(f, "{}", v),
            Expr::Neg(ref x) => match *x.body {
                Expr::Constant(_) | Expr::Variable(_)  => write!(f, "-{}", x),
                _  => write!(f, "-({})", x),
            },
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            //Expr::Mul(ref a, ref b) => write_with_parens(a, "x", b, f),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<'a> Neg for Function {
    type Output = Function;
    fn neg(self) -> Function {
        Function {
            output: None,
            variables: self.variables.clone(),
            body: box Expr::Neg(self),
        }
    }
}

impl<'a> Add for Function {
    type Output = Function;
    fn add(self, other: Function) -> Function {
        let vars1 = self.variables.clone();
        let vars2 = other.variables.clone();

        Function {
            output: None,
            variables: vars1.union(&vars2).cloned().collect(),
            body: box Expr::Add(self, other),
        }
    }
}

//impl<'a> Mul for Function {
    //type Output = Function;
    //fn add(self, other: Function) -> Function {
        //let vars1 = self.variables.clone();
        //let vars2 = other.variables.clone();

        //Function {
            //output: None,
            //variables: vars1.union(&vars2).cloned().collect(),
            //body: box Expr::Mul(self, other),
        //}
    //}
//}

pub fn variable<'a>(s: &str, dims: Vec<i32>) -> Function {
    let mut vars = HashSet::new();
    vars.insert(String::from(s));
    Function {
        output: None,
        variables: vars,
        body: box Expr::Variable(Variable {
                name: String::from(s),
                dims: dims,
        })
    }

}

pub fn scalar<'a>(x: f32) -> Function {
    Function {
        output: Some(Constant::Scalar(x)),
        variables: HashSet::new(),
        body: box Expr::Constant(Constant::Scalar(x)), 
    }
}


pub fn grad<'a>(f: &Function, var: &str) -> Constant {
    match f.variables.contains::<str>(&var) {
        false => Constant::Scalar(0.),
        true => match *f.body { 
            Expr::Constant(ref c) => copy_and_fill(c, 0.), 
            Expr::Variable(ref v) => new_constant(v.dims.clone(), 1.),
            Expr::Neg(ref f) => -grad(&f, var),
            Expr::Add(ref f1, ref f2) => grad(&f1, var) + grad(&f2, var),
            //Expr::Mul(ref f1, ref f2) => grad(&f1, var) * grad(&f2, var),
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


pub fn eval<'a>(f: &Function, args: &HashMap<&str, Constant>) -> Option<Constant> {
    match *f.body { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Variable(ref var) => args.get::<str>(&var.name).map(|x| x.clone()),
        Expr::Neg(ref f) => eval(&f, args).map(|x| -x),
        Expr::Add(ref f1, ref f2) => apply_to_branches(&|x, y| x + y, args, f1, f2),
        //Expr::Add(ref f1, ref f2) => 
        //Expr::Mul(ref f1, ref f2) => apply_to_branches(|x, y| x * y, f1 f2),
    }
}

pub fn assign_outputs(f: &mut Function, args: &HashMap<&str, Constant>) {
    f.output = match *(&mut *f.body) { 
        Expr::Constant(ref x) => Some(x.clone()),
        Expr::Variable(ref var) => args.get::<str>(&var.name).map(|x| x.clone()),
        Expr::Neg(ref mut f1) => {
            assign_outputs(f1, args);
            f1.output.clone().map(|x| -x)
        }
        Expr::Add(ref mut f1, ref mut f2) => {
            assign_outputs(f1, args);
            assign_outputs(f2, args);
            apply_to_branches(&|x, y| x + y, args, f1, f2)
        } 
    }
}
