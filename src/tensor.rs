use std::ops::{Add, Sub};

#[cfg(feature = "random_generation")]
use rand::{rngs::ThreadRng, Rng};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Tensor<const LEN: usize>(pub(crate) [f64; LEN]);

impl<const LEN: usize> Default for Tensor<LEN> {
    fn default() -> Self {
        Self([0.; LEN])
    }
}

impl<const LEN: usize> Tensor<LEN> {
    pub fn new(array: [f64; LEN]) -> Self {
        Self(array)
    }
    pub fn sum(&self) -> f64 {
        self.0.iter().sum()
    }
    pub fn sum_product(&self, rhs: &Self) -> f64 {
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(lhs, rhs)| rhs * lhs)
            .sum()
    }
    /// Finds the squared length of the tensor
    pub fn length_sqrd(&self) -> f64 {
        self.0.iter().fold(0., |current, new| current + new.powi(2))
    }
    pub fn length(&self) -> f64 {
        self.length_sqrd().sqrt()
    }
    pub fn set(&mut self, index: usize, value: f64) {
        self.0[index] = value;
    }
}

#[cfg(feature = "random_generation")]
impl<const LEN: usize> Tensor<LEN> {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        let mut output = Self::default();

        for i in output.0.iter_mut() {
            *i = rng.gen();
        }

        output
    }
}

impl<const LEN: usize> Add for Tensor<LEN> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (lhs, rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *lhs += rhs
        }

        self
    }
}

impl<const LEN: usize> Sub for Tensor<LEN> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (lhs, rhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *lhs -= rhs
        }

        self
    }
}
