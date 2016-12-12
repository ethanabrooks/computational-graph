use std::fmt;
use function::datatypes::{Function, Expr};
use std::ops::Deref;

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


                    //Constant::Matrix(m) => {
                        //let mut i = 0;
                        //let mut repr;
                        //loop {
                            //repr = format!("m{}", i);
                            //if !matrices.contains(repr) {
                                //break;
                            //}
                            //i += 1;
                        //}
                        //matrices.insert(repr, m);
                    //},
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
            Expr::Sq(ref x) => match *x.body.clone() {
                Expr::Constant(_) | Expr::Input(_)  => write!(f, "{}²", x),
                _                                   => write!(f, "({})²", x),
            },
            Expr::Abs(ref x)        => write!(f, "|{}|", x),
            Expr::Signum(ref x)     => write!(f, "sign({})", x),
            Expr::Sigmoid(ref x)    => write!(f, "σ({})", x),
            Expr::Tanh(ref x)    => write!(f, "tanh({})", x),
            Expr::Add(ref a, ref b) => write_with_parens(a, "+", b, f),
            Expr::Sub(ref a, ref b) => write_with_parens(a, "-", b, f),
            Expr::Mul(ref a, ref b) => write_with_parens(a, "⚬", b, f),
            Expr::Dot(ref a, ref b, trans1, trans2) => {
                let t_symb1 = if trans1 { "ᵀ" } else { "" };
                let t_symb2 = if trans2 { "ᵀ" } else { "" };
                match *a.body.clone() {
                    Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) =>
                        match *b.body.clone() {
                            Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                                write!(f, "〈{}{}, {}{}〉", a, t_symb1, b, t_symb2),
                            _ => write!(f, "〈{}{}, ({}){}〉", a, t_symb1, b, t_symb2),
                        },
                    _  => match *b.body.clone() {
                            Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) => 
                                write!(f, "〈({}){}, {}{}〉",  a, t_symb1, b, t_symb2),
                            _ => write!(f, "〈({}){}, ({}){}〉",  a, t_symb1, b, t_symb2),
                    }
                }
            }
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self.body {
            Expr::Param(ref p) => write!(f, "{}≔{}", p.name, self.unwrap_value().deref()),
            _ => write!(f, "{}", self.body),
        }
    }
}

