pub mod activation;
mod cost;
mod ghost;
mod linear;
mod matrix;
mod parameters;

pub use cost::cost;
pub use ghost::GhostToken;
pub use linear::LinearLayer;
pub use matrix::Matrix;
pub use parameters::Parameters;

pub trait Model<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn forward(
        &self,
        token: &GhostToken<'id>,
        matrix: &Matrix<INPUT_LEN, 1>,
    ) -> Matrix<OUTPUT_LEN, 1>;
}

pub trait AsParameters<'id> {
    fn as_parameters(&self, token: &mut GhostToken<'id>);
}
