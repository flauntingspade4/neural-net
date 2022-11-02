use crate::{Backpropagation, Differentiable, Matrix};

pub trait LossFunction<const LEN: usize>: Differentiable<LEN, LEN> {
    fn loss_function(self, matrix: Matrix<LEN, 1>) -> f64;
}

pub struct MSEloss<const OUTPUT_LEN: usize> {
    pub expected: Matrix<OUTPUT_LEN, 1>,
}

impl<const OUTPUT_LEN: usize> LossFunction<OUTPUT_LEN> for MSEloss<OUTPUT_LEN> {
    fn loss_function(self, matrix: Matrix<OUTPUT_LEN, 1>) -> f64 {
        let difference = matrix - self.expected;

        difference.elements().fold(0., |old, new| old + new.powi(2)) / (OUTPUT_LEN as f64)
    }
}

impl<const OUTPUT_LEN: usize> Differentiable<OUTPUT_LEN, OUTPUT_LEN> for MSEloss<OUTPUT_LEN> {
    fn calculate_grads(
        &mut self,
        mut backpropagation: Backpropagation<OUTPUT_LEN>,
        matrix: Matrix<OUTPUT_LEN, 1>,
    ) -> Backpropagation<OUTPUT_LEN> {
        for (element, (actual, expected)) in backpropagation
            .running_total
            .elements_mut()
            .zip(matrix.elements().zip(self.expected.elements()))
        {
            *element = 2. * (*actual - *expected) / OUTPUT_LEN as f64;
        }

        backpropagation
    }
}
