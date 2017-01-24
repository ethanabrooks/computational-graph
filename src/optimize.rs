use function::{Function, Expr};
use constant::Constant;
//use ops::{negate};
//{mul_assign, sq_ref, signum_ref, add_assign, sub_assign, 
          //sigmoid_assign, signum_assign, tanh_assign, sq_assign, 
          //abs_assign, negate, one_minus};
use std::collections::HashMap;
//use std::io::{Write, stderr};
use std::ops::{Deref, DerefMut};

macro_rules! exec {
    [($result:expr) = ($arg1:expr) - ($arg2:expr)] => {
        ($result).sub_assign(($arg1), ($arg2));
    };
    [($result:expr) = ($arg1:expr) * ($arg2:expr)] => {
        ($result).mul_assign($arg1, $arg2);
    };
    [($result:expr) = ($arg1:expr) + ($arg2:expr)] => {
        ($result).add_assign($arg1, $arg2);
    };
    [($result:expr) -= ($arg:expr)] => {
        ($result).sub($arg);
    };
    [($result:expr) *= ($arg:expr)] => {
        ($result).mul($arg);
    };
    [($result:expr) = ($arg:expr)] => {
        $result.copy($arg);
    };
    [($result:expr) = -($arg:expr)] => {
        exec!(($result) *= ($arg))
    };
}

macro_rules! value {
    ($f:expr) => { $f.value().deref() }
}

macro_rules! placeholder {
    ($f:expr) => { $f.placeholder(0).deref_mut() };
    ($f:expr, $i:expr) => { $f.placeholder($i).deref_mut() };
}

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
            //Expr::Sq(ref f)           => f.eval(args) * f.eval(args),
            //Expr::Abs(ref f)          => f.eval(args).abs(),
            //Expr::Sigmoid(ref f)      => f.eval(args).sigmoid(),
            //Expr::Tanh(ref f)         => f.eval(args).tanh(),
            //Expr::Signum(ref f)       => f.eval(args).signum(),
            Expr::Add(ref f1, ref f2) => f1.eval(args) + f2.eval(args),
            Expr::Sub(ref f1, ref f2) => f1.eval(args) - f2.eval(args),
            Expr::Mul(ref f1, ref f2) => f1.eval(args) * f2.eval(args),
            //Expr::Dot(ref f1, ref f2, trans1, trans2) =>
                //Constant::dot(&f1.eval(args), &f2.eval(args), trans1, trans2)
        }
    }

    pub fn grad(&self, param: &str) -> Function {
        if self.params().contains::<str>(&param) {
            match *self.body() {
                Expr::Neg(ref f)          => -f.grad(param),
                //Expr::Sq(ref f)           => &f.grad(param) * f,
                //Expr::Abs(ref f)          => signum_ref(f) * f.grad(param),
                //Expr::Signum(_)           => panic!("signum is nondifferentiable"),
                //Expr::Sigmoid(ref f)      =>
                    //f.grad(param) * (self.clone() * (&Function::scalar(1.) - self)),
                //Expr::Tanh(ref f)         => 
                    //f.grad(param) * (Function::scalar(1.) - sq_ref(self)),
                Expr::Add(ref f1, ref f2) => f1.grad(param) + f2.grad(param),
                Expr::Sub(ref f1, ref f2) => f1.grad(param) - f2.grad(param),
                Expr::Mul(ref f1, ref f2) => &f1.grad(param) * f2 +
                                             &f2.grad(param) * f1,
                //Expr::Dot(_, _, _, _) => panic!("not implemented"),
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
            //if (i + 1) % print_freq  == 0 {
                //println!("{}", self.value().deref());
            //}
        }
    }

    //#[allow(dead_code)]
    //pub fn maximize(&self,
                    //args: &HashMap<&str, Constant>,
                    //learn_rate: f32,
                    //iters: i32,
                    //print_freq: i32) {
        //(-self).minimize(args, learn_rate, iters, print_freq);
    //}

    //fn assign1(&self, child: &Function, args: &HashMap<&str, Constant>,
               //mutation: &Fn(&mut Constant)) {
        //child.assign_values(args);
        //if self.get_value().is_none() {
            //self.set_value(Constant::empty_like(child.unwrap_value().deref()))
        //}
        //self.mutate_value(&|x| {
            //x.copy(child.unwrap_value().deref());
            //mutation(x)
        //})
    //}

    //fn assign2(&self, child1: &Function, child2: &Function,
               //args: &HashMap<&str, Constant>, mutation: &Fn(&mut Constant, &Constant)) {
        //child1.assign_values(args);
        //child2.assign_values(args);


        //let mut refmut = self.value_mut();
        //let value = refmut.deref_mut();
        //value.copy(child1.value().deref());
        //mutation(value, child2.value().deref());
        ////self.mutate_value(&|x| {
            ////x.copy(child1.unwrap_value().deref());
            ////mutation(x, child2.unwrap_value().deref())
        ////});
    //}
    pub fn assign_values(&self, args: &HashMap<&str, Constant>) {
        // assign final value to outputs
        match *self.body() {
            Expr::Constant(_) | Expr::Param(_) => return,
            //Expr::Input(ref i) =>
                //self.set_value(args.get::<str>(&i.name).expect("missing arg").clone()),
                //// TODO: avoid clone?
            Expr::Neg(ref f) => {
                f.assign_values(args);
                exec!((self.value_mut()) = -(value!(f)));
            }
            //Expr::Sq(ref f) => self.assign1(f, args, &sq_assign),
            //Expr::Abs(ref f) => self.assign1(f, args, &abs_assign),
            //Expr::Signum(ref f) => {
                //writeln!(&mut stderr(), "WARN: Signum is non-differentiable.
                //Running `backprop` on this function will cause an error").unwrap();
                //self.assign1(f, args, &signum_assign);
            //}
            //Expr::Sigmoid(ref f) => self.assign1(f, args, &sigmoid_assign),
            //Expr::Tanh(ref f) => self.assign1(f, args, &tanh_assign),
            Expr::Add(ref f1, ref f2) => {
                assign_values(f1, f2, args);
                exec![(self.value_mut()) = (value!(f1)) + (value!(f2))]
            }
            Expr::Sub(ref f1, ref f2) => {
                assign_values(f1, f2, args);
                exec![(self.value_mut()) = (value!(f1)) - (value!(f2))]
            }
            Expr::Mul(ref f1, ref f2) => {
                assign_values(f1, f2, args);
                exec![(self.value_mut()) = (value!(f1)) * (value!(f2))]
            }
            //Expr::Dot(ref f1, ref f2, trans1, trans2) => {
                //f1.assign_values(args);
                //f2.assign_values(args);
                //let val1 = f1.unwrap_value();
                //let val2 = f2.unwrap_value();
                //if self.get_value().is_none() {
                    //self.set_value(Constant::empty_for_dot(val1.deref(), val2.deref(),
                                                          //trans1, trans2));
                //}
                //self.mutate_value(&|x| x.assign_dot(val1.deref(), val2.deref(),
                                                    //trans1, trans2));
            //}
        }
    }

    fn backprop(&self, error: &mut Constant, learn_rate: f32) {
        //self.maybe_alloc_placeholders(error);
        if self.params().is_empty() { return; }
        match *self.body() {
            Expr::Param(_) => {
                //*error *= Constant::Scalar(learn_rate);
                //let mut value = self.unwrap_value_mut();
                //let self_value = self.value_mut();
                exec![(self.value_mut()) = (error) * (&Constant::Scalar(learn_rate))];
                //self.mutate_value(&|x| x -= error);
            }
            Expr::Neg(ref f) => {
                error.negate();
                f.backprop(error, learn_rate)
            }
            //Expr::Sq(ref f) => {
                //mul_assign(error, f.unwrap_value().deref());
                //f.backprop(error, learn_rate)
            //}
            //Expr::Abs(ref f) => {
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(f.unwrap_value().deref()); // out
                    //signum_assign(x);                 // signum(out)
                    //mul_assign(x, error);             // error * signum(out)
                //});
                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            //}
            //Expr::Signum(_) => panic!("sign is not differentiable"),
            //Expr::Sigmoid(ref f) => {
                //let val = self.unwrap_value();
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(val.deref());        // out
                    //one_minus(x);               // 1 - out
                    //mul_assign(x, val.deref()); // out * (1 - out)
                    //mul_assign(x, error);       // error * out * (1 - out)
                //});

                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            //}
            //Expr::Tanh(ref f) => {
                //let val = self.unwrap_value();
                //self.mutate_placeholder(0, &|x| {
                    //x.copy(val.deref());        // out
                    //sq_assign(x);               // out^2
                    //one_minus(x);               // 1 - out^2
                    //mul_assign(x, error);       // error * (1 - out^2)
                //});

                //f.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
            //}
            Expr::Add(ref f1, ref f2) => {
                exec![(placeholder!(self)) = (error)];

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(self), learn_rate);
            }
            Expr::Sub(ref f1, ref f2) => {
                exec![(placeholder!(self)) = -(error)];

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(self), learn_rate);
            }
            Expr::Mul(ref f1, ref f2) => {
                exec![(placeholder!(self)) = (value!(f1)) * (error)];
                exec![(error) *= (value!(f2))];

                f1.backprop(error, learn_rate);
                f2.backprop(placeholder!(self), learn_rate);
            }
            //Expr::Dot(ref f1, ref f2, trans1, trans2) => {
                //// placeholder[0]: dot(error, f2.T)
                //self.mutate_placeholder(0, &|x| x.assign_dot(&error, &f2.unwrap_value(),
                                                             //false, !trans2));
                //// placeholder[1]: dot(f1.T, error)
                //self.mutate_placeholder(1, &|x| x.assign_dot(&f1.unwrap_value(),
                                                             //&error, !trans1, false));

                //f1.backprop(self.get_placeholder(0).deref_mut(), learn_rate);
                //f2.backprop(self.get_placeholder(1).deref_mut(), learn_rate);
            //}
            Expr::Constant(_)
                //| Expr::Input(_) 
                => return,
        }
    }

}

fn assign_values(f1: &Function, f2: &Function, args: &HashMap<&str, Constant>) {
    f1.assign_values(args);
    f2.assign_values(args);
}

