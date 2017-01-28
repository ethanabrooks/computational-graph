macro_rules! exec {
    [($result:expr) = dot(($arg1:expr) T=$t1:expr, ($arg2:expr) T=$t2:expr)] => {{
        $result.assign_dot($arg1, $t1, $arg2, $t2)
    }};
    [($result:expr) = ($arg1:expr) + ($arg2:expr)] => {{
        $result.copy($arg1);
        $result.add_assign($arg2);
    }};
    [($result:expr) = ($arg1:expr) - ($arg2:expr)] => {{
        $result.copy($arg1);
        $result.sub_assign($arg2);
    }};
    [($result:expr) = ($arg1:expr) * ($arg2:expr)] => {{
        $result.copy($arg1);
        $result.mul_assign($arg2);
    }};
    [($result:expr) = ($arg1:expr) $op:ident ($arg2:expr)] => {{
        $result.copy($arg1);
        ($result).op(($arg1), ($arg2));
    }};
    [($result:expr) -= ($arg:expr)] => {
        ($result).sub_assign($arg);
    };
    [($result:expr) *= ($arg:expr)] => {
        ($result).mul_assign($arg);
    };
    [($result:expr) = 1 - ($arg:expr)] => {
        $result.copy($arg);
        one_minus($result);
    };
    [($result:expr) = -($arg:expr)] => {
        $result.copy($arg);
        negate($result);
    };
    [($result:expr) = $op:ident ($arg:expr)] => {
        $result.copy($arg);
        $op($result);
    };
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

