use crate::{matrix::Matrix, Model};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

#[derive(Debug, Clone, PartialEq)]
pub struct LinearLayer<const INPUT_LEN: usize, const OUTPUT_LEN: usize> {
    weights: Matrix<OUTPUT_LEN, INPUT_LEN>,
    biases: Matrix<OUTPUT_LEN, 1>,
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize> Default
    for LinearLayer<INPUT_LEN, OUTPUT_LEN>
{
    fn default() -> Self {
        Self {
            weights: Matrix::new_zeroed(),
            biases: Matrix::new_zeroed(),
        }
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize> Model<INPUT_LEN, OUTPUT_LEN>
    for LinearLayer<INPUT_LEN, OUTPUT_LEN>
{
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1> {
        self.weights * *matrix + self.biases
    }
}

#[cfg(feature = "random_generation")]
impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize> LinearLayer<INPUT_LEN, OUTPUT_LEN> {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            // nodes: [(); OUTPUT_LEN].map(|_| LinearNode::random_new(rng)),
            weights: Matrix::random_new(rng),
            biases: Matrix::random_new(rng),
        }
    }
}

/*#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
struct LinearNode<const INPUT_LEN: usize> {
    weights: Tensor<INPUT_LEN>,
    bias: f64,
}

#[cfg(feature = "random_generation")]
impl<const INPUT_LEN: usize> LinearNode<INPUT_LEN> {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            weights: Tensor::random_new(rng),
            bias: rng.gen(),
        }
    }
}*/
