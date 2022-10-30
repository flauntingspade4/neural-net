use std::marker::PhantomData;

use crate::{activation::ActivationFunction, matrix::Matrix, Model};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

pub struct LinearLayer<
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    T: ActivationFunction<OUTPUT_LEN>,
> {
    weights: Matrix<OUTPUT_LEN, INPUT_LEN>,
    biases: Matrix<OUTPUT_LEN, 1>,
    activation: PhantomData<T>,
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    pub fn new(weights: Matrix<OUTPUT_LEN, INPUT_LEN>, biases: Matrix<OUTPUT_LEN, 1>) -> Self {
        Self {
            weights: weights,
            biases: biases,
            activation: PhantomData,
        }
    }
}

/*impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> Debug
    for (&GhostToken<'id>, LinearLayer<'id, INPUT_LEN, OUTPUT_LEN>)
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> Clone
    for (&GhostToken<'id>, LinearLayer<'id, INPUT_LEN, OUTPUT_LEN>)
{
    fn clone(&self) -> Self {
        todo!()
    }
}*/

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>> Default
    for LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    fn default() -> Self {
        Self {
            weights: Matrix::new_zeroed(),
            biases: Matrix::new_zeroed(),
            activation: PhantomData,
        }
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    Model<INPUT_LEN, OUTPUT_LEN> for LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1> {
        let pre_activation = self.weights * *matrix + self.biases;

        T::activate(pre_activation)
    }
}

#[cfg(feature = "random_generation")]
impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    LinearLayer<INPUT_LEN, OUTPUT_LEN, T>
{
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            weights: Matrix::random_new(rng),
            biases: Matrix::random_new(rng),
            activation: PhantomData,
        }
    }
}
