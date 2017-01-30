macro_rules! exec {
    [($result:expr) = dot(($arg1:expr) T=$t1:expr, ($arg2:expr) T=$t2:expr)] => {{
        $result.equals_dot($arg1, $t1, $arg2, $t2)
    }};
    [($result:expr) = ($arg1:expr) + ($arg2:expr)] => {{
        $result.equals_add($arg1, $arg2);
    }};
    [($result:expr) = ($arg1:expr) - ($arg2:expr)] => {{
        $result.equals_sub($arg1, $arg2);
    }};
    [($result:expr) = ($arg1:expr) * ($arg2:expr)] => {{
        $result.equals_mul($arg1, $arg2);
    }};
    [($result:expr) -= ($arg:expr)] => {
        ($result).sub_assign($arg);
    };
    [($result:expr) *= ($arg:expr)] => {
        ($result).mul_assign($arg);
    };
    [($result:expr) = 1 - ($arg:expr)] => {
        $result.equals_one_minus($arg);
    };
    [($result:expr) = -($arg:expr)] => {
        $result.equals_neg($arg);
    };
    [($result:expr) = sq($arg:expr)] => {{
        $result.equals_sq($arg);
    }};
    [($result:expr) = abs($arg:expr)] => {{
        $result.equals_abs($arg);
    }};
    [($result:expr) = signum($arg:expr)] => {{
        $result.equals_signum($arg);
    }};
    [($result:expr) = sigmoid($arg:expr)] => {{
        $result.equals_sigmoid($arg);
    }};
    [($result:expr) = tanh($arg:expr)] => {{
        $result.equals_tanh($arg);
    }};
}

macro_rules! value {
    ($f:expr) => { $f.value().deref() }
}

macro_rules! value_mut {
    ($f:expr) => { $f.value_mut().deref_mut() }
}

macro_rules! dot {
    ($f1:expr, $f2:expr) => {
        dot($f1, false, $f2, false)
    };
    (($f1:expr)^T, $f2:expr) => {
        dot($f1, true, $f2, false)
    };
    ($f1:expr, ($f2:expr)^T) => {
        dot($f1, false, $f2, true)
    };
    (($f1:expr)^T, ($f2:expr)^T) => {
        dot($f1, true, $f2, true)
    };
}

