use function::ops::{dot, sigmoid, tanh};
use function::datatypes::{Expr, Function, Constant};

// TODO: add this to datatypes.rs so that this file can be moved out of function
fn check_params(functions: Vec<&Function>) {
    for function in functions {
        match *function.body {
            Expr::Param(_) => {},
            _ => panic!("{} must be a param", &function),
        }
    }
}

#[allow(non_snake_case)]
pub fn lstm_custom_params(input: Vec<Constant>, 
            Wi: Function, 
            Ui: Function, 
            bi: Function, 
            Wc: Function, 
            Uc: Function, 
            bc: Function, 
            Wf: Function, 
            Uf: Function, 
            bf: Function, 
            Wo: Function, 
            Uo: Function, 
            Vo: Function, 
            bo: Function, 
            C: Function,
            h: Function,
            ) -> Function {
    check_params(vec![&Wi, &Ui, &bi, &Wc, &Uc, &bc,
                      &Wf, &Uf, &bf, &Wo, &Uo, &Vo, &bo, &C, &h]);
    match &input[..] {
        &[] => C,
        &[ref head, ref tail..] => {
            let x = Function::constant(head.clone());

            // The LSTM equations
            let i = sigmoid(&(dot(&x, &Wi) + (&dot(&h, &Ui) + &bi)));
            let c = tanh(&(dot(&x, &Wc) + (&dot(&h, &Uc) + &bc)));
            let f = sigmoid(&(dot(&x, &Wf) + (&dot(&h, &Uf) + &bf)));
            let C_new = i * c + (&f * &C);
            let o = sigmoid(&(dot(&x, &Wo) + dot(&h, &Uo) + (&dot(&C_new, &Vo) + &bo)));
            let h_new = o * tanh(&C);

            lstm_custom_params(tail.to_vec(), Wi, Ui, bi, Wc, Uc, bc, Wf, Uf, bf,
                                              Wo, Uo, Vo, bo, C_new, h_new)
        }
    }
}

fn get_input_dim(inputs: &Vec<Constant>) -> u32 {
    let mut dim = None;
    for input in inputs {
        match dim {
            Some(d) => assert!(d == input.width(), "inputs have inconsistent dimension"),
            None    => dim = Some(input.width()),
        }
    }
    dim.unwrap()
}

#[allow(non_snake_case, dead_code)]
pub fn lstm(inputs: Vec<Constant>) -> Function {
    let d = get_input_dim(&inputs);
    let Wi = Function::random_param("Wi", vec![d, d], -1., 1.); 
    let Ui = Function::random_param("Ui", vec![d, d], -1., 1.); 
    let bi = Function::random_param("bi", vec![d, d], -1., 1.); 
    let Wc = Function::random_param("Wc", vec![d, d], -1., 1.); 
    let Uc = Function::random_param("Uc", vec![d, d], -1., 1.); 
    let bc = Function::random_param("bc", vec![d, d], -1., 1.); 
    let Wf = Function::random_param("Wf", vec![d, d], -1., 1.); 
    let Uf = Function::random_param("Uf", vec![d, d], -1., 1.); 
    let bf = Function::random_param("bf", vec![d, d], -1., 1.); 
    let Wo = Function::random_param("Wo", vec![d, d], -1., 1.); 
    let Uo = Function::random_param("Uo", vec![d, d], -1., 1.); 
    let Vo = Function::random_param("Vo", vec![d, d], -1., 1.); 
    let bo = Function::random_param("bo", vec![d, d], -1., 1.); 
    let C = Function::random_param("C", vec![d, d], -1., 1.);
    let h = Function::random_param("h", vec![d, d], -1., 1.);

    lstm_custom_params(inputs, Wi, Ui, bi, Wc, Uc, bc, 
                       Wf, Uf, bf, Wo, Uo, Vo, bo, C, h)
}

#[allow(non_snake_case, dead_code)]
pub fn rnn(input: Vec<Constant>, hidden_state: Function, bias: Function) -> Function {
    match *hidden_state.body {
        Expr::Param(_) => { 
            match &input[..] {
                &[] => hidden_state,
                &[ref head, ref tail..] => {
                    let x = Function::constant(head.clone());
                    sigmoid(&(&dot(&x, &hidden_state) + &bias)) 
                        + rnn(tail.to_vec(), hidden_state, bias)
                }
            }
        }
        _ => panic!("rnn needs to take a param for hidden state"),
    }
}
