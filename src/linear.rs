use std::marker::PhantomData;

use crate::{
    activation::ActivationFunction, matrix::Matrix, Backpropagation, Differentiable, Model,
};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

pub struct LinearLayer<
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    T: ActivationFunction<OUTPUT_LEN>,
> {
    weights: Matrix<OUTPUT_LEN, INPUT_LEN>,
    biases: Matrix<OUTPUT_LEN, 1>,
    weights_grad: Matrix<OUTPUT_LEN, INPUT_LEN>,
    biases_grad: Matrix<OUTPUT_LEN, 1>,
    activation: PhantomData<T>,
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    pub fn new(weights: Matrix<OUTPUT_LEN, INPUT_LEN>, biases: Matrix<OUTPUT_LEN, 1>) -> Self {
        Self {
            weights,
            biases,
            weights_grad: Matrix::new_zeroed(),
            biases_grad: Matrix::new_zeroed(),
            activation: PhantomData,
        }
    }

    pub fn new_zeroed() -> Self {
        Self::new(Matrix::new_zeroed(), Matrix::new_zeroed())
    }

    #[cfg(feature = "random_generation")]
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self::new(Matrix::random_new(rng), Matrix::random_new(rng))
    }

    pub fn zero_grad(&mut self) {
        self.weights_grad = Matrix::new_zeroed();
        self.biases_grad = Matrix::new_zeroed();
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>> Default
    for LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    fn default() -> Self {
        Self::new_zeroed()
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    Model<INPUT_LEN, OUTPUT_LEN> for LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1> {
        T::activate(self.weights * *matrix + self.biases)
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    Differentiable<INPUT_LEN, OUTPUT_LEN> for LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    fn calculate_grads(
        &mut self,
        backpropagation: Backpropagation<OUTPUT_LEN>,
        matrix: Matrix<OUTPUT_LEN, 1>,
    ) -> Backpropagation<INPUT_LEN> {
        for ((total_derivative, previous_layer_activation), (i, bias_grad)) in backpropagation
            .running_total
            .elements()
            .zip(matrix.elements())
            .zip(self.biases_grad.elements_mut().enumerate())
        {
            for input_i in 0..INPUT_LEN {
                unsafe {
                    self.weights_grad
                        .set_unchecked((i, input_i), previous_layer_activation * *total_derivative);
                }
            }

            *bias_grad += *total_derivative;
        }

        backpropagation.transform(self.weights)
    }
}
