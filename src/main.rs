mod function; 
mod constant; 
use function::scalar;
use function::variable;
use function::Variable;
use function::eval;
use function::grad;
use std::collections::HashMap;

fn main() {
    let a = scalar(2.);
    let x = variable("x");
    println!("{:#?}", a);
    println!("{:#?}", x);

    let f = a + x;
    println!("{:#?}", f);

    let mut args = HashMap::new();
    args.insert(String::from("x"), constant::Constant::scalar(3.));

    //println!("{:#?}", eval(*f, &args));
    println!("{:#?}", grad(*f, &Variable { 
        name: String::from("x"), 
        gradient: None 
    }));
}
