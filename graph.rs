use std::fmt;
use std::ops::{Neg, Add, Mul};
use constant::{Constant, copy_and_fill, new_constant, new_matrix};
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;

#[derive(Debug)]
enum Expr<'a> {
    Constant(f32),
    Add(&'a Node<'a>, &'a Node<'a>),
}

#[derive(Debug)]
pub struct Node<'a> {
    output: RefCell<Option<f32>>,
    pub variables: HashSet<String>,
    body: Expr<'a>,
}

impl<'a> Add for &'a Node<'a> {
    type Output = Node<'a>;
    fn add(self, other: &'a Node<'a>) -> Node<'a> {
        let vars1 = self.variables.clone();
        let vars2 = other.variables.clone();

        Node {
            output: RefCell::new(None),
            variables: vars1.union(&vars2).cloned().collect(),
            body: Expr::Add(self, other),
        }
    }
}

pub fn node<'a>(x: f32) -> Node<'a> {
    Node {
        output: RefCell::new(Some(x)),
        variables: HashSet::new(),
        body: Expr::Constant(x), 
    }
}

