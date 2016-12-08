use std::fmt;
use std::io::{Write, stderr};
use function::datatypes::{Function, Expr};

// helper function for fmt
fn write_with_parens(a: &Function, 
                     operator: &str,
                     b: &Function,  
                     f: &mut fmt::Formatter) -> fmt::Result {
    match *a.body.clone() {
        Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) =>
            match *b.body.clone() {
                Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                     write!(f, "{} {} {}", a, operator, b),
                _ => write!(f, "{} {} ({})", a, operator, b),
            },
        _  => match *b.body.clone() {
                Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                     write!(f, "({}) {} {}", a, operator, b),
                _ => write!(f, "({}) {} ({})", a, operator, b),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Expr::Constant(ref c) => write!(f, "{}", c), 
            Expr::Input(ref i) => write!(f, "{}", i.name),
            Expr::Param(ref p) => write!(f, "{}", p.name),
            Expr::Neg(ref x) => match *x.body.clone() {
                Expr::Constant(_) | Expr::Input(_)  => write!(f, "-{}", x),
                _                                   => write!(f, "-({})", x),
            },
            Expr::Abs(ref x)        => write!(f, "|{}|", x),
            Expr::Signum(ref x)     => write!(f, "sign({})", x),
            Expr::Sigmoid(ref x)    => write!(f, "sigmoid({})", x),
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            Expr::Sub(ref a, ref b) => write_with_parens(a, "-", b, f),
            Expr::Mul(ref a, ref b) => write_with_parens(a, "×", b, f),
            Expr::Dot(ref a, ref b) => write_with_parens(a, "", b, f),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self.body {
            Expr::Param(ref p) => write!(f, "{}≔{}", p.name, self.unwrap_value().clone()),
            _                  => write!(f, "{}", *self.body.clone())
        }
    }
}

