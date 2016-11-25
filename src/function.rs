use std::fmt;
use std::ops::Neg;
use std::ops::Add;
use constant::Constant;
use constant::copy_and_fill;
use constant::new_constant;
use std::collections::HashMap;
use std::collections::HashSet;
use std::cell::RefCell;

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
    //sub(Expr, Expr),
}

#[derive(Debug)]
pub struct Function {
    output: RefCell<Option<Constant>>,
    pub variables: HashSet<String>,
    body: Box<Expr>,
}

impl<'a> fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.body)
    }
}

impl<'a> fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Expr::Constant(ref c) => write!(f, "{}", c), 
            &Expr::Variable(ref v) => write!(f, "{}", v),
            &Expr::Neg(ref x) => match x.body {
                Expr::Constant(_) | Expr::Variable(_)  => write!(f, "-{}", x),
                _  => write!(f, "-({})", x),
            },
            &Expr::Add(ref a, ref b) => 
                match a.body {
                    Expr::Constant(_) | Expr::Variable(_) =>
                        match b.body {
                            Expr::Constant(_) | Expr::Variable(_) => 
                                write!(f, "{} + {}", a, b),
                            _ => write!(f, "{} + ({})", a, b),
                        },
                    _  =>
                        match b.body {
                            Expr::Constant(_) | Expr::Variable(_) => 
                                write!(f, "({}) + {}", a, b),
                            _ => write!(f, "({}) + ({})", a, b),
                        }
                }
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
            body: Expr::Neg(self),
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
            body: Expr::Add(self, other),
        }
    }
}

pub fn variable<'a>(s: &str, dims: Vec<i32>) -> Function {
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

pub fn scalar<'a>(x: f32) -> Function {
    Function {
        output: Some(Constant::Scalar(x)),
        variables: HashSet::new(),
        body: Expr::Constant(Constant::Scalar(x)), 
    }
}


pub fn grad<'a>(f: &Function, var: &str) -> Constant {
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

pub fn eval<'a>(f: &Function, args: &HashMap<&str, Constant>) -> Option<Constant> {
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

//pub fn assign_outputs_<'a>(f: &mut Function) { //, args: &HashMap<&str, f32>) {
    //match f.body { 
          //Expr::Constant(_) => println!("{:?}", f.body),
          //Expr::Neg(ref mut ff) => 
              //assign_outputs_(ff),
    //}
//}

pub fn assign_outputs<'a>(f: &mut Function, args: &HashMap<&str, Constant>) {
    match f.body { 
        Expr::Constant(_) => f.output = eval(f, args),
        Expr::Variable(_) => f.output = eval(f, args),
        Expr::Neg(ref mut f1) => {
            assign_outputs(f1, args);
            f.output = f1.output.clone().map(|x| -x);
        }
        Expr::Add(ref mut f1, ref mut f2) => {
            //assign_outputs(f1, args);
            //assign_outputs(f2, args);
            match (f1.output.clone(), f2.output.clone()) {
                (Some(x1), Some(x2)) => *f.output.borrow_mut() = Some(x1 + x2),
                _                    => f.output = None,
            };
        }
    }
}
