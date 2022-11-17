use core::ops::{Add, Mul, Sub};

use rand::{rngs::ThreadRng, Rng};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<const COLUMNS: usize, const ROWS: usize> {
    inner: [[f64; COLUMNS]; ROWS],
}

impl<const COLUMNS: usize, const ROWS: usize> Matrix<COLUMNS, ROWS> {
    /// Initializes a new matrix of 0s.
    #[must_use]
    pub const fn new_zeroed() -> Self {
        Self {
            inner: [[0.; COLUMNS]; ROWS],
        }
    }
    #[cfg(feature = "random_generation")]
    #[must_use]
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self { inner: rng.gen() }
    }
    #[must_use]
    pub const fn from_arrays(inner: [[f64; COLUMNS]; ROWS]) -> Self {
        Self { inner }
    }
    pub unsafe fn set_unchecked(&mut self, (x, y): (usize, usize), value: f64) {
        *self.inner.get_unchecked_mut(x).get_unchecked_mut(y) = value;
    }
    pub fn elements(&self) -> impl Iterator<Item = &f64> {
        self.inner.iter().flatten()
    }
    pub fn elements_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.inner.iter_mut().flatten()
    }
    pub fn columns(&self) -> impl Iterator<Item = &[f64; COLUMNS]> {
        self.inner.iter()
    }
}

impl<const COLUMNS: usize> Matrix<COLUMNS, 1> {
    /// Finds the squared length of the matrix
    pub fn length_sqrd(&self) -> f64 {
        self.inner[0]
            .iter()
            .fold(0., |current, new| current + new.powi(2))
    }
    pub fn length(&self) -> f64 {
        self.length_sqrd().sqrt()
    }
}

impl<const COLUMNS: usize, const ROWS: usize> Default for Matrix<COLUMNS, ROWS> {
    fn default() -> Self {
        Self::new_zeroed()
    }
}

impl<const COLUMNS: usize, const ROWS: usize> Add for Matrix<COLUMNS, ROWS> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for columns in self.inner.iter_mut().zip(rhs.inner.iter()) {
            for row in columns.0.iter_mut().zip(columns.1.iter()) {
                *row.0 += *row.1;
            }
        }
        self
    }
}

impl<const COLUMNS: usize, const ROWS: usize> Sub for Matrix<COLUMNS, ROWS> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for columns in self.inner.iter_mut().zip(rhs.inner.iter()) {
            for row in columns.0.iter_mut().zip(columns.1.iter()) {
                *row.0 -= *row.1;
            }
        }
        self
    }
}

impl<const COLUMNS: usize, const ROWS: usize, const NEW_ROWS: usize> Mul<Matrix<ROWS, NEW_ROWS>>
    for Matrix<COLUMNS, ROWS>
{
    type Output = Matrix<COLUMNS, NEW_ROWS>;

    fn mul(self, rhs: Matrix<ROWS, NEW_ROWS>) -> Self::Output {
        let mut matrix = Matrix::new_zeroed();

        for column in 0..COLUMNS {
            for row in 0..NEW_ROWS {
                let mut value = 0.;

                for i in 0..ROWS {
                    value += self.inner[i][column] * rhs.inner[row][i];
                }

                matrix.inner[row][column] = value;
            }
        }

        matrix
    }
}

#[test]
fn matrix_multiplication() {
    let lhs = Matrix::from_arrays([[0., 1., 2.], [0., 2., 3.]]);

    let rhs = Matrix::from_arrays([[0., 1.], [1., 2.], [2., 3.]]);

    let expected = Matrix::from_arrays([[0.0, 2.0, 3.0], [0.0, 5.0, 8.0], [0.0, 8.0, 13.0]]);

    assert_eq!(expected, lhs * rhs);
}
