#![warn(clippy::pedantic, clippy::nursery, clippy::perf)]
#![no_std]

pub mod activation;
pub mod cost;
mod linear;
mod matrix;
pub mod optim;

pub use linear::LinearLayer;
pub use matrix::Matrix;

pub trait Model<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1>;
}

pub trait Differentiable<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    fn calculate_grads(
        &mut self,
        backpropagation: Backpropagation<OUTPUT_LEN>,
        activations: Matrix<INPUT_LEN, 1>,
    ) -> Backpropagation<INPUT_LEN>;
}

#[derive(Default)]
pub struct Backpropagation<const LEN: usize> {
    pub total_derivatives: Matrix<LEN, 1>,
}

impl<const LEN: usize> Backpropagation<LEN> {
    pub const fn new(cost_output: Matrix<LEN, 1>) -> Self {
        Self {
            total_derivatives: cost_output,
        }
    }

    pub fn transform<const NEW_LEN: usize>(
        self,
        weights: Matrix<LEN, NEW_LEN>,
    ) -> Backpropagation<NEW_LEN> {
        let mut back: [f64; NEW_LEN] = [0.; NEW_LEN];

        for (new_neuron, weights) in back.iter_mut().zip(weights.columns()) {
            for (neuron, weight) in self.total_derivatives.elements().zip(weights.iter()) {
                *new_neuron += neuron * *weight;
            }
        }

        Backpropagation::new(Matrix::from_arrays([back]))
    }
}
