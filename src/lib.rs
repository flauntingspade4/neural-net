pub mod activation;
mod cost;
mod linear;
mod matrix;

pub use cost::cost;
pub use linear::LinearLayer;
pub use matrix::Matrix;

pub trait Model<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1>;
}
