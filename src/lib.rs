mod cost;
mod linear;
mod matrix;
mod relu;

pub use cost::cost;
pub use linear::LinearLayer;
pub use matrix::Matrix;
pub use relu::Relu;

pub trait Model<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1>;
}
