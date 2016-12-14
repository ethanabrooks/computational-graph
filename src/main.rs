#![feature(advanced_slice_patterns, slice_patterns)]

extern crate libc;
extern crate lifeguard;
extern crate rand;

mod function; 
mod constant; 

#[allow(unused_imports)]
use function::{sq, dot, abs, sigmoid, rnn, Function};
#[allow(unused_imports)]
use constant::{Matrix, Constant};
#[allow(unused_imports)]
use constant::Constant::Scalar;
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::ops::Deref;

extern { fn init_cublas(); }

//static POOL: Vec<Matrix> = vec![];

fn main() {
    unsafe { init_cublas() };
    let args = HashMap::new();
    println!("f: {}", &f);
}
