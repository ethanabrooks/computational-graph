use function::{Function, Expr};
use constant::Constant;
use ops::{one_minus, negate, sq, abs, tanh, sigmoid, signum};
//{mul_assign, sq_ref, signum_ref, add_assign, sub_assign, 
          //sigmoid_assign, signum_assign, tanh_assign, sq_assign, 
          //abs_assign, negate, one_minus};
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
            let mut error = self.value_mut().copy_and_fill(1.);
            self.backprop(&mut error, learn_rate);
            if (i + 1) % print_freq  == 0 {
                println!("before print");
                //println!("{}", self);
                self.value().print();
            }
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
                exec![(value_mut!(self)) = (value!(f1)) + (value!(f2))]
            }
            Expr::Sub(ref f1, ref f2) => {
                f1.assign_values(args);
                f2.assign_values(args);
                exec![(value_mut!(self)) = (value!(f1)) - (value!(f2))];

                print!("f1: ");
                f1.print();
                print!("f2: ");
                f2.print();
                print!("value: ");
                self.value().print();
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
                //print!("\n\nerror: "); 
                //error.print();
                *error *= Constant::Scalar(learn_rate);
                //print!("error after multiplication with learn_rate: "); 
                //error.print();
                //print!("value: "); 
                //self.value().print();
                exec![(self.value_mut()) -= (error)];
                //print!("value after subtraction of error: "); 
                //self.value().print();
                //println!("\n");
            }
            Expr::Neg(ref f) => {
                negate(error);
                f.backprop(error, learn_rate)
            }
            Expr::Sq(ref f) => {
                exec![(error) *= (&Constant::Scalar(2.))];  // TODO: is this right??
                exec![(error) *= (value!(f))];
                f.backprop(error, learn_rate)
            }
            Expr::Abs(ref f) => {
                //print!("value: ");
                //self.value().print();
                exec![(placeholder!()) = signum(value!(f))];
                //print!("signum(value): ");
                //(placeholder!()).print();
                exec![(placeholder!()) *= (error)];
                f.backprop(placeholder!(), learn_rate);
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(f.unwrap_value().deref()); // out
                    //signum_assign(x);                 // signum(out)
                    //mul_assign(x, error);             // error * signum(out)
                //});
                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            }
            Expr::Signum(_) => panic!("sign is not differentiable"),
            Expr::Sigmoid(ref f) => {
                exec![(placeholder!()) = 1 - (value!(self))];
                exec![(placeholder!()) *= (value!(self))]; 
                exec![(error) *= (placeholder!())];
                f.backprop(placeholder!(), learn_rate);
            }
                //let val = self.unwrap_value();
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(val.deref());        // out
                    //one_minus(x);               // 1 - out
                    //mul_assign(x, val.deref()); // out * (1 - out)
                    //mul_assign(x, error);       // error * out * (1 - out)
                //});

                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            Expr::Tanh(ref f) => {
                exec![(placeholder!()) = sq(value!(self))];
                one_minus(placeholder!());
                exec![(error) *= (placeholder!())];
                f.backprop(placeholder!(), learn_rate);
            }
                //let val = self.unwrap_value();
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(val.deref());        // out
                    //sq_assign(x);               // out^2
                    //one_minus(x);               // 1 - out^2
                    //mul_assign(x, error);       // error * (1 - out^2)
                //});

                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            Expr::Add(ref f1, ref f2) => {
                placeholder!().copy(error);

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                exec![(placeholder!()) = -(error)];

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                exec![(placeholder!()) = (value!(f1)) * (error)];
                exec![(error) *= (value!(f2))];

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(), learn_rate);
            }
            Expr::Dot(ref f1, t1, ref f2, t2) => {
                exec![(placeholder!(0)) = dot((error) T=false, (value!(f2)) T=!t2)];
                exec![(placeholder!(1)) = dot((value!(f1)) T=!t1, (error) T=false)];

                f1.backprop(placeholder!(0), learn_rate);
                f2.backprop(placeholder!(1), learn_rate);

                //self.mutate_placeholder(0, &|x| x.assign_dot(&error, &f2.unwrap_value(),
                                                             //false, !trans2));
                //self.mutate_placeholder(1, &|x| x.assign_dot(&f1.unwrap_value(),
                                                             //&error, !trans1, false));
                //f1.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
                //f2.backprop(self.get_placeholder(1).deref_mut(), learn_rate);
            }
            Expr::Constant(_)
                //| Expr::Input(_) 
                => return,
        }
    }

}
