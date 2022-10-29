mod cost;
mod linear;
mod matrix;
mod relu;
mod tensor;

pub use cost::cost;
pub use linear::LinearLayer;
pub use matrix::Matrix;
pub use relu::Relu;
pub use tensor::Tensor;

pub trait Model<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn forward(&self, tensor: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1>;
}
