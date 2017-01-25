use ops::dot;
use function::Function;
use constant::Constant;

// TODO: add this to datatypes.rs so that this file can be moved out of function
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
    Function::check_params(vec![&Wi, &Ui, &bi, &Wc, &Uc, &bc,
                           &Wf, &Uf, &bf, &Wo, &Uo, &Vo, &bo]);
    match &input[..] {
        &[] => C,
        &[ref head, ref tail..] => {
            let x = Function::constant(head.clone());

            // The LSTM equations
            let i = (dot!(&x, &Wi) + (&dot!(&h, &Ui) + &bi)).sigmoid();
            let c = (dot!(&x, &Wc) + (&dot!(&h, &Uc) + &bc)).tanh();
            let f = (dot!(&x, &Wf) + (&dot!(&h, &Uf) + &bf)).sigmoid();
            let C_new = i * c + (&f * &C);
            let o = (dot!(&x, &Wo) + dot!(&h, &Uo) + (&dot!(&C_new, &Vo) + &bo)).sigmoid();
            let h_new = o * C.tanh();

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

    lstm_custom_params(inputs, 
        Function::random_param("Wi", vec![d, d], -1., 1.),
        Function::random_param("Ui", vec![d, d], -1., 1.),
        Function::random_param("bi", vec![d, d], -1., 1.),
        Function::random_param("Wc", vec![d, d], -1., 1.),
        Function::random_param("Uc", vec![d, d], -1., 1.),
        Function::random_param("bc", vec![d, d], -1., 1.),
        Function::random_param("Wf", vec![d, d], -1., 1.),
        Function::random_param("Uf", vec![d, d], -1., 1.),
        Function::random_param("bf", vec![d, d], -1., 1.),
        Function::random_param("Wo", vec![d, d], -1., 1.),
        Function::random_param("Uo", vec![d, d], -1., 1.),
        Function::random_param("Vo", vec![d, d], -1., 1.),
        Function::random_param("bo", vec![d, d], -1., 1.),
        Function::random_param("C", vec![d, d], -1., 1.),
        Function::random_param("h", vec![d, d], -1., 1.)
    )
}


#[allow(non_snake_case, dead_code)]
pub fn rnn(inputs: Vec<&Constant>, hidden_state: Function, bias: Function) -> Function {
    match &inputs[..] {
        &[] => hidden_state,
        &[ref head, ref tail..] => {
            let x = Function::constant((*head).clone());
            (&dot!(&x, &hidden_state) + &bias).sigmoid()
                + rnn(tail.to_vec(), hidden_state, bias)
        }
    }
}
