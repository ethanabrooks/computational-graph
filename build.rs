extern crate gcc;

fn main() {
    gcc::Config::new().file("src/matrix.c").compile("libmatrix.a");
}
