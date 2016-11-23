mod function; 
mod constant; 
use function::variable;
use function::scalar;
use function::eval;
use function::grad;
use constant::Constant::Scalar;
use std::collections::HashMap;

fn main() {
    let a = scalar(2.);
    let x = variable("x");
    let a = &a;
    let x = &x;
    println!("{:#?}", a);
    println!("{:#?}", x);

    let f = a + x;
    println!("{:#?}", f);

    let mut args = HashMap::new();
    args.insert("x", Scalar(3.));

    println!("{:#?}", eval(&f, &args));
    println!("{:#?}", grad(&f, "x"));
}


/*TODO:
 * add dim field to Var
 * get grad to handle matrices
 */
