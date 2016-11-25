mod function; 
mod constant; 
use function::variable;
use function::scalar;
use function::eval;
use function::grad;
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let mut a = scalar(2.);
    let mut x = variable("x", vec![]);
    //let a = &mut a;
    //let x = &mut x;
    println!("{}", a);
    println!("{}", x);

    let f = &mut x + &mut a;
    //println!("{}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));

    //println!("{:?}", eval(&f, &args));
    //println!("{:?}", grad(&f, "x"));
}

