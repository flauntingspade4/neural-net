use crate::{
    ghost::{GhostCell, GhostToken},
    matrix::Matrix,
    Model,
};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

pub struct LinearLayer<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    weights: GhostCell<'id, Matrix<OUTPUT_LEN, INPUT_LEN>>,
    biases: GhostCell<'id, Matrix<OUTPUT_LEN, 1>>,
}

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> LinearLayer<'id, INPUT_LEN, OUTPUT_LEN> {
    pub fn new(weights: Matrix<OUTPUT_LEN, INPUT_LEN>, biases: Matrix<OUTPUT_LEN, 1>) -> Self {
        Self {
            weights: GhostCell::new(weights),
            biases: GhostCell::new(biases),
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

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> Default
    for LinearLayer<'id, INPUT_LEN, OUTPUT_LEN>
{
    fn default() -> Self {
        Self {
            weights: GhostCell::new(Matrix::new_zeroed()),
            biases: GhostCell::new(Matrix::new_zeroed()),
        }
    }
}

impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> Model<'id, INPUT_LEN, OUTPUT_LEN>
    for LinearLayer<'id, INPUT_LEN, OUTPUT_LEN>
{
    fn forward(
        &self,
        token: &GhostToken<'id>,
        matrix: &Matrix<INPUT_LEN, 1>,
    ) -> Matrix<OUTPUT_LEN, 1> {
        *self.weights.borrow(token) * *matrix + *self.biases.borrow(token)
    }
}

#[cfg(feature = "random_generation")]
impl<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize> LinearLayer<'id, INPUT_LEN, OUTPUT_LEN> {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            weights: GhostCell::new(Matrix::random_new(rng)),
            biases: GhostCell::new(Matrix::random_new(rng)),
        }
    }
}
