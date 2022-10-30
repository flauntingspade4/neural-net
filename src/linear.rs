use std::marker::PhantomData;

use crate::{
    activation::ActivationFunction,
    ghost::{GhostCell, GhostToken},
    matrix::Matrix,
    Model,
};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

pub struct LinearLayer<
    'id,
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    T: ActivationFunction<OUTPUT_LEN>,
> {
    weights: GhostCell<'id, Matrix<OUTPUT_LEN, INPUT_LEN>>,
    biases: GhostCell<'id, Matrix<OUTPUT_LEN, 1>>,
    activation: PhantomData<T>,
}

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    LinearLayer<'id, INPUT_LEN, OUTPUT_LEN, T>
{
    pub fn new(weights: Matrix<OUTPUT_LEN, INPUT_LEN>, biases: Matrix<OUTPUT_LEN, 1>) -> Self {
        Self {
            weights: GhostCell::new(weights),
            biases: GhostCell::new(biases),
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

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    Default for LinearLayer<'id, INPUT_LEN, OUTPUT_LEN, T>
{
    fn default() -> Self {
        Self {
            weights: GhostCell::new(Matrix::new_zeroed()),
            biases: GhostCell::new(Matrix::new_zeroed()),
            activation: PhantomData,
        }
    }
}

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    Model<'id, INPUT_LEN, OUTPUT_LEN> for LinearLayer<'id, INPUT_LEN, OUTPUT_LEN, T>
{
    fn forward(
        &self,
        token: &GhostToken<'id>,
        matrix: &Matrix<INPUT_LEN, 1>,
    ) -> Matrix<OUTPUT_LEN, 1> {
        let pre_activation = *self.weights.borrow(token) * *matrix + *self.biases.borrow(token);

        T::activate(pre_activation)
    }
}

#[cfg(feature = "random_generation")]
impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize, T: ActivationFunction<OUTPUT_LEN>>
    LinearLayer<'id, INPUT_LEN, OUTPUT_LEN, T>
{
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            weights: GhostCell::new(Matrix::random_new(rng)),
            biases: GhostCell::new(Matrix::random_new(rng)),
            activation: PhantomData,
        }
    }
}
