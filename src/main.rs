mod function; 
mod constant; 
use function::scalar;
use function::variable;

fn main() {
    let a = scalar(2.);
    let x = variable("x");
    println!("{:#?}", a);
    println!("{:#?}", x);

    let f = a + x;
    println!("{:#?}", f);


}
