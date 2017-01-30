use function::{Function, Expr};
use constant::Constant;
use std::collections::HashMap;
use std::io::{Write, stderr};
use std::ops::{Deref, DerefMut};

impl Function {
    pub fn eval(&self, args: &HashMap<&str, Constant>) -> Constant {
        match *self.body() {
            Expr::Constant(ref x) => x.clone(),
            //Expr::Input(ref i) => {
                //match args.get::<str>(&i.name) {
                    //Some(val) => val.clone(),
                    //None => panic!("`args` is missing {}. Content of `args`: \n{:#?}",
                                   //&i.name, args),
                //}
            //}
            Expr::Param(_)            => self.value().clone(),
            Expr::Neg(ref f)          => -f.eval(args),
            Expr::Sq(ref f)           => f.eval(args) * f.eval(args),
            Expr::Abs(ref f)          => f.eval(args).abs(),
            Expr::Sigmoid(ref f)      => f.eval(args).sigmoid(),
            Expr::Tanh(ref f)         => f.eval(args).tanh(),
            Expr::Signum(ref f)       => f.eval(args).signum(),
            Expr::Add(ref f1, ref f2) => f1.eval(args) + f2.eval(args),
            Expr::Sub(ref f1, ref f2) => f1.eval(args) - f2.eval(args),
            Expr::Mul(ref f1, ref f2) => f1.eval(args) * f2.eval(args),
            Expr::Dot(ref f1, trans1, ref f2, trans2) =>
                Constant::dot(&f1.eval(args), trans1, &f2.eval(args), trans2)
        }
    }

    pub fn grad(&self, param: &str) -> Function {
        if self.params().contains::<str>(&param) {
            match *self.body() {
                Expr::Neg(ref f)          => -f.grad(param),
                Expr::Sq(ref f)           => &f.grad(param) * f,
                Expr::Abs(ref f)          => (f.clone()).signum() * f.grad(param),
                Expr::Signum(_)           => panic!("signum is nondifferentiable"),
                Expr::Sigmoid(ref f)      =>
                    f.grad(param) * (self.clone() * (&Function::scalar(1.) - self)),
                Expr::Tanh(ref f)         => 
                    f.grad(param) * (Function::scalar(1.) - self.clone().sq()),
                Expr::Add(ref f1, ref f2) => f1.grad(param) + f2.grad(param),
                Expr::Sub(ref f1, ref f2) => f1.grad(param) - f2.grad(param),
                Expr::Mul(ref f1, ref f2) => &f1.grad(param) * f2 +
                                             &f2.grad(param) * f1,
                Expr::Dot(_, _, _, _) => panic!("not implemented"),
                Expr::Param(_) => Function::constant(self.value_mut()
                                                         .copy_and_fill(1.)),
                Expr::Constant(_) 
                    //| Expr::Input(_) 
                    => panic!("should never reach here"),
            }
        } else {
            Function::scalar(0.)
        }
    }

    #[allow(dead_code)]
    pub fn minimize(&self, 
                    args: &HashMap<&str, Constant>,
                    learn_rate: f32,
                    iters: i32,
                    print_freq: i32) {
        for i in 0..iters {
            self.assign_values(&args);
            if (i + 1) % print_freq  == 0 {
                self.value().print();
            }
            let mut error = self.value_mut().copy_and_fill(1.);
            self.backprop(&mut error, learn_rate);
        }
    }

    #[allow(dead_code)]
    pub fn maximize(&self,
                    args: &HashMap<&str, Constant>,
                    learn_rate: f32,
                    iters: i32,
                    print_freq: i32) {
        (-self).minimize(args, learn_rate, iters, print_freq);
    }

    // assign final value to outputs
    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        match *self.body() {
            Expr::Constant(_) | Expr::Param(_) => return,
            //Expr::Input(ref i) =>
                //self.set_value(args.get::<str>(&i.name).expect("missing arg").clone()),
            Expr::Neg(ref f) => {
                f.assign_values(args);
                exec![(value_mut!(self)) = -(value!(f))];
            }
            Expr::Sq(ref f) => {
                f.assign_values(args);
                exec![(value_mut!(self)) = sq(value!(f))];
            }
            Expr::Abs(ref f) => {
                f.assign_values(args);
                exec![(value_mut!(self)) = abs(value!(f))];
            }
            Expr::Signum(ref f) => {
                f.assign_values(args);
                writeln!(&mut stderr(), "WARN: Signum is non-differentiable.
                //Running `backprop` on this function will cause an error").unwrap();
                exec![(value_mut!(self)) = signum(value!(f))];
            }
            Expr::Sigmoid(ref f) => {
                f.assign_values(args);
                exec!((value_mut!(self)) = sigmoid(value!(f)));
            }
            Expr::Tanh(ref f) => {
                f.assign_values(args);
                exec![(value_mut!(self)) = tanh(value!(f))];
            }
            Expr::Add(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                exec![(value_mut!(self)) = (value!(f1)) + (value!(f2))];
            }
            Expr::Sub(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                exec![(value_mut!(self)) = (value!(f1)) - (value!(f2))];
            }
            Expr::Mul(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                exec![(value_mut!(self)) = (value!(f1)) * (value!(f2))];
            }
            Expr::Dot(ref f1, t1, ref f2, t2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                exec![(value_mut!(self)) = dot((value!(f1)) T=t1, (value!(f2)) T=t2)];
                //self.mutate_value(&|x| x.assign_dot(val1.deref(), val2.deref(),
                                                    //trans1, trans2));
            }
        }
    }

    // TODO:make this private
    pub fn backprop(&self, error: &mut Constant, learn_rate: f32) {

        macro_rules! placeholder {
            () => { self.placeholder(0).deref_mut() };
            ($i:expr) => { self.placeholder($i).deref_mut() };
        }

        if self.params().is_empty() { return; }
        match *self.body() {
            Expr::Param(_) => {
                *error *= Constant::Scalar(learn_rate); // possibly matrix * scalar
                exec![(self.value_mut()) -= (error)]; // possibly scalar -= matrix
            }
            Expr::Neg(ref f) => {
                error.assign_neg();
                f.backprop(error, learn_rate)
            }
            Expr::Sq(ref f) => {
                exec![(error) *= (&Constant::Scalar(2.))];
                exec![(error) *= (value!(f))];  // possibly scalar *= matrix
                f.backprop(error, learn_rate)
            }
            Expr::Abs(ref f) => {
                exec![(placeholder!()) = signum(value!(f))];
                exec![(placeholder!()) *= (error)];
                f.backprop(placeholder!(), learn_rate);
            }
            Expr::Signum(_) => panic!("sign is not differentiable"),
            Expr::Sigmoid(ref f) => {
                exec![(placeholder!()) = 1 - (value!(self))];
                exec![(placeholder!()) *= (value!(self))]; 
                exec![(error) *= (placeholder!())];
                f.backprop(placeholder!(), learn_rate);
            }
            Expr::Tanh(ref f) => {
                exec![(placeholder!()) = sq(value!(self))];
                placeholder!().assign_one_minus();
                exec![(error) *= (placeholder!())];
                f.backprop(placeholder!(), learn_rate);
            }
            Expr::Add(ref f1, ref f2) => {
                placeholder!().absorb(error); 
                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                exec![(placeholder!()) = -(error)];
                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                exec![(placeholder!()) = (value!(f1)) * (error)]; // if scalar = scalar * matrix, first reduce matrix to scalar.
                exec![(error) *= (value!(f2))];
                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Dot(ref f1, t1, ref f2, t2) => {
                exec![(placeholder!(0)) = dot((error) T=false, (value!(f2)) T=!t2)];
                exec![(placeholder!(1)) = dot((value!(f1)) T=!t1, (error) T=false)];
                f1.backprop(placeholder!(0), learn_rate);
                f2.backprop(placeholder!(1), learn_rate);
            }
            Expr::Constant(_)
                //| Expr::Input(_) 
                => return,
        }
    }

}
