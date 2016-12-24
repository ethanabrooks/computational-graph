use function::datatypes::{Function, Expr, Constant};
use function::{mul_assign, sq_ref, signum_ref,
                    add_assign, sub_assign, sigmoid_assign, signum_assign, 
                    tanh_assign, sq_assign, abs_assign, negate, one_minus};
use std::collections::HashMap;
use std::io::{Write, stderr};
use std::ops::{Deref, DerefMut};

impl Function {
    pub fn eval(&self, args: &HashMap<&str, Constant>) -> Constant {
        match *self.body {
            Expr::Constant(ref x) => x.clone(),
            Expr::Input(ref i) => {
                match args.get::<str>(&i.name) {
                    Some(val) => val.clone(),
                    None => panic!("`args` is missing {}. Content of `args`: \n{:#?}",
                                   &i.name, args),
                }
            }
            Expr::Param(_)            => self.unwrap_value().clone(),
            Expr::Neg(ref f)          => -f.eval(args),
            Expr::Sq(ref f)           => f.eval(args) * f.eval(args),
            Expr::Abs(ref f)          => f.eval(args).abs(),
            Expr::Sigmoid(ref f)      => f.eval(args).sigmoid(),
            Expr::Tanh(ref f)         => f.eval(args).tanh(),
            Expr::Signum(ref f)       => f.eval(args).signum(),
            Expr::Add(ref f1, ref f2) => f1.eval(args) + f2.eval(args),
            Expr::Sub(ref f1, ref f2) => f1.eval(args) - f2.eval(args),
            Expr::Mul(ref f1, ref f2) => f1.eval(args) * f2.eval(args),
            Expr::Dot(ref f1, ref f2, trans1, trans2) =>
                Constant::dot(&f1.eval(args), &f2.eval(args), trans1, trans2)
        }
    }

    pub fn grad(&self, param: &str) -> Function {
        if self.params.contains::<str>(&param) {
            match *self.body {
                Expr::Neg(ref f)          => -f.grad(param),
                Expr::Sq(ref f)           => &f.grad(param) * f,
                Expr::Abs(ref f)          => signum_ref(f) * f.grad(param),
                Expr::Signum(_)           => panic!("signum is nondifferentiable"),
                Expr::Sigmoid(ref f)      =>
                    f.grad(param) * (self.clone() * (&Function::scalar(1.) - self)),
                Expr::Tanh(ref f)         => 
                    f.grad(param) * (Function::scalar(1.) - sq_ref(self)),
                Expr::Add(ref f1, ref f2) => f1.grad(param) + f2.grad(param),
                Expr::Sub(ref f1, ref f2) => f1.grad(param) - f2.grad(param),
                Expr::Mul(ref f1, ref f2) => &f1.grad(param) * f2 +
                                             &f2.grad(param) * f1,
                Expr::Dot(_, _, _, _) => panic!("not implemented"),
                Expr::Param(_) => Function::constant(self.unwrap_value()
                                                       .copy_and_fill(1.)),
                Expr::Constant(_) | Expr::Input(_) => panic!("should never reach here"),
            }
        } else {
            Function::scalar(0.)
        }
    }

    #[allow(dead_code)]
    pub fn minimize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
            println!("0");
        for i in 0..iters {
            println!("1");
            self.assign_values(&args);
            println!("2");
            let mut error = self.unwrap_value().copy_and_fill(1.);
            println!("4");
            self.backprop(&mut error, learn_rate);
            println!("5");
            if i % 100 == 0 {
                println!("{}", self.unwrap_value().clone());
            }
            println!("6");
        }
    }

    #[allow(dead_code)]
    pub fn maximize(&self, args: &HashMap<&str, Constant>, learn_rate: f32, iters: i32) {
        (-self).minimize(args, learn_rate, iters);
    }
}

impl Function {
    fn assign1(&self, child: &Function, args: &HashMap<&str, Constant>,
               mutation: &Fn(&mut Constant)) {
        child.assign_values(args);
        if self.get_value().is_none() {
            self.set_value(Constant::empty_like(child.unwrap_value().deref()))
        }
        self.mutate_value(&|x| {
            x.copy(child.unwrap_value().deref());
            mutation(x)
        })
    }

    fn assign2(&self, child1: &Function, child2: &Function,
               args: &HashMap<&str, Constant>, mutation: &Fn(&mut Constant, &Constant)) {
        child1.assign_values(args);
        child2.assign_values(args);
        if self.get_value().is_none() {
            self.set_value(Constant::empty_like(child1.unwrap_value().deref()))
        }
        self.mutate_value(&|x| {
            x.copy(child1.unwrap_value().deref());
            mutation(x, child2.unwrap_value().deref())
        });
    }

    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        // assign final value to outputs
        match *self.body {
            Expr::Constant(_) | Expr::Param(_) => assert!(self.get_value().is_some(),
                "Constants and Params must always have a value"),
            Expr::Input(ref i) =>
                self.set_value(args.get::<str>(&i.name).expect("missing arg").clone()),
                // TODO: avoid clone?
            Expr::Neg(ref f) => self.assign1(f, args, &negate),
            Expr::Sq(ref f) => self.assign1(f, args, &sq_assign),
            Expr::Abs(ref f) => self.assign1(f, args, &abs_assign),
            Expr::Signum(ref f) => {
                writeln!(&mut stderr(), "WARN: Signum is non-differentiable.
                Running `backprop` on this function will cause an error").unwrap();
                self.assign1(f, args, &signum_assign);
            }
            Expr::Sigmoid(ref f) => self.assign1(f, args, &sigmoid_assign),
            Expr::Tanh(ref f) => self.assign1(f, args, &tanh_assign),
            Expr::Add(ref f1, ref f2) => self.assign2(f1, f2, args, &add_assign),
            Expr::Sub(ref f1, ref f2) => self.assign2(f1, f2, args, &sub_assign),
            Expr::Mul(ref f1, ref f2) => self.assign2(f1, f2, args, &mul_assign),
            Expr::Dot(ref f1, ref f2, trans1, trans2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                let val1 = f1.unwrap_value();
                let val2 = f2.unwrap_value();
                if self.get_value().is_none() {
                    self.set_value(Constant::empty_for_dot(val1.deref(), val2.deref(),
                                                          trans1, trans2));
                }
                self.mutate_value(&|x| x.assign_dot(val1.deref(), val2.deref(),
                                                    trans1, trans2));
            }
        }
    }

    fn backprop(&self, error: &mut Constant, learn_rate: f32) {
        self.maybe_alloc_placeholders(error);
        if self.params.is_empty() { return; }
        match *self.body {
            Expr::Param(_) => {
                *error *= Constant::Scalar(learn_rate);
                self.mutate_value(&|x| sub_assign(x, &error));
            }
            Expr::Neg(ref f) => {
                negate(error);
                f.backprop(error, learn_rate)
            }
            Expr::Sq(ref f) => {
                mul_assign(error, f.unwrap_value().deref());
                f.backprop(error, learn_rate)
            }
            Expr::Abs(ref f) => {
                self.mutate_placeholder(0, &|x| {
                    x.copy(f.unwrap_value().deref()); // out
                    signum_assign(x);                 // signum(out)
                    mul_assign(x, error);             // error * signum(out)
                });
                f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Signum(_) => panic!("sign is not differentiable"),
            Expr::Sigmoid(ref f) => {
                let val = self.unwrap_value();
                self.mutate_placeholder(0, &|x| {
                    x.copy(val.deref());        // out
                    one_minus(x);               // 1 - out
                    mul_assign(x, val.deref()); // out * (1 - out)
                    mul_assign(x, error);       // error * out * (1 - out)
                });

                f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Tanh(ref f) => {
                let val = self.unwrap_value();
                self.mutate_placeholder(0, &|x| {
                    x.copy(val.deref());        // out
                    sq_assign(x);               // out^2
                    one_minus(x);               // 1 - out^2
                    mul_assign(x, error);       // error * (1 - out^2)
                });

                f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Add(ref f1, ref f2) => {
                self.mutate_placeholder(0, &|x| x.copy(error));
                f1.backprop(error, learn_rate);
                f2.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                self.mutate_placeholder(0, &|x| {
                    x.copy(error); // error
                    negate(x);     // -error
                });

                f1.backprop(error, learn_rate);
                f2.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                self.mutate_placeholder(0, &|x| {
                    x.copy(error);
                    mul_assign(x, &f1.unwrap_value().deref()); // error * f1
                });
                mul_assign(error, &f2.unwrap_value().deref()); // error * f2

                f1.backprop(error, learn_rate);
                f2.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Dot(ref f1, ref f2, trans1, trans2) => {
                // placeholder[0]: dot(error, f2.T)
                self.mutate_placeholder(0, &|x| x.assign_dot(&error, &f2.unwrap_value(),
                                                             false, !trans2));
                // placeholder[1]: dot(f1.T, error)
                self.mutate_placeholder(1, &|x| x.assign_dot(&f1.unwrap_value(),
                                                             &error, !trans1, false));

                f1.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
                f2.backprop(self.get_placeholder(1).deref_mut(), learn_rate);
            }
            Expr::Constant(_)| Expr::Input(_) => return,
        }
    }

    fn maybe_alloc_placeholders(&self, error: &Constant) {
        match *self.body {
            Expr::Constant(_) | Expr::Input(_) | Expr::Param(_) |
            Expr::Neg(_)      | Expr::Sq(_)    | Expr::Signum(_) => return,
            Expr::Sigmoid(ref f) | Expr::Tanh(ref f)   | Expr::Add(ref f, _) |
            Expr::Sub(ref f, _)  | Expr::Mul(ref f, _) | Expr::Abs(ref f) => {
                if self.num_placeholders() < 1 {
                    self.alloc_placeholders(
                        vec![Constant::empty_like(f.unwrap_value().deref())]);


                }
            }
            Expr::Dot(ref f1, ref f2, trans1, trans2) =>
                if self.num_placeholders() < 2 {
                    self.alloc_placeholders(
                        vec![Constant::empty_for_dot(
                                &error, &f2.unwrap_value().deref(), false, !trans2),
                            Constant::empty_for_dot(
                                &f1.unwrap_value().deref(), &error, !trans1, false)])
                }
        };
    }

}

