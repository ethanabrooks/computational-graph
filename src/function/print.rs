use std::fmt;
use function::datatypes::{Function, Expr, Constant, Matrix};
use std::ops::Deref;

extern {
    fn download_matrix(m1: *const Matrix, m2: *mut f32);
}

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


// TODO: make this a macro
fn fmt_(c: &Constant, f: &mut fmt::Formatter) -> fmt::Result {
    match *c {
        Constant::Scalar(x) => write!(f, "{}", x),
        Constant::Matrix(ref src) => {
            let dst: *mut f32;
            unsafe { download_matrix(src.borrow(), dst) };
            let mut result;

            let h = src.height() - 1;
            result = if h == 0 { write!(f, "\n{:^2}", "[") }
            else               { write!(f, "\n{:^2}", "⎡")
            };
            if result.is_err() { return result }

            for i in 0..src.height() {

                for j in 0..src.width() {
                    result = write!(f, "{:^10.3}", unsafe { 
                        *dst.offset((j * src.height() + i) as isize) });
                    if result.is_err() { return result }
                }

                result = if h == 0           { write!(f, "{:^2}\n", "]") }

                else     if i == 0 && h == 1 { write!(f, "{:^2}\n{:^2}", "⎤", "⎣" ) }

                else     if i == h - 1       { write!(f, "{:^2}\n{:^2}", "⎥", "⎣") }

                else     if i == 0           { write!(f, "{:^2}\n{:^2}", "⎤", "⎢") }

                else     if i == h           { write!(f, "{:^2}\n", "⎦") } 

                else                         { write!(f, "{:^2}\n{:^2}", "⎥", "⎢") };

                if result.is_err() { return result }
            }
            unsafe { drop(Box::from_raw(dst)); }
            result
        }
    }
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}


