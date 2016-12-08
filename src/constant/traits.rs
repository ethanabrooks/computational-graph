
impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let mut m = empty_like(self);
        unsafe { copy_matrix(self as *const Matrix, &mut m) };
        m
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { free_matrix(self as *mut Matrix) };
    }
} 

fn size(matrix: &Matrix) -> i32 {
    matrix.height * matrix.width
}

fn fmt_(c: &Constant, f: &mut fmt::Formatter) -> fmt::Result {
    match *c {
        Constant::Scalar(x) => write!(f, "{}", x),
        Constant::Matrix(ref src) => {
            let mut dst = Vec::with_capacity(size(src) as usize);
            unsafe { download_matrix(src, dst.as_mut_ptr()) };
            let mut result;

            let h = src.height - 1;
            result = if h == 0 { write!(f, "\n{:^2}", "[") }
            else               { write!(f, "\n{:^2}", "⎡")
            };
            if result.is_err() { return result }

            for i in 0..src.height {

                for j in 0..src.width {
                    result = write!(f, "{:^10.3}", unsafe {
                        *dst.as_ptr().offset((i * src.width + j) as isize) 
                    });
                    if result.is_err() { return result }
                }

                result = if h == 0           { write!(f, "{:^2}\n", "]") }

                else     if i == 0 && h == 1 { write!(f, "{:^2}\n{:^2}", "⎤", "⎣" ) }

                else     if i == h - 1       { write!(f, "{:^2}\n{:^2}", "⎥", "⎣") }

                else     if i == 0           { write!(f, "{:^2}\n{:^2}", "⎤", "⎢") }

                else     if i == h           { write!(f, "{:^2}\n", "⎦") } 

                else                         { write!(f, "{:^2}\n{:^2}", "⎥", "⎢") };

                if result.is_err() { return result }
            }
            result
        }
    }
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt_(self, f) }
}

