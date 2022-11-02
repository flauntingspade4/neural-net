use crate::Matrix;

pub trait ActivationFunction<const OUTPUT_LEN: usize> {
    fn activate(matrix: Matrix<OUTPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1>;
}

pub struct ReLU;

impl<const OUTPUT_LEN: usize> ActivationFunction<OUTPUT_LEN> for ReLU {
    fn activate(mut matrix: Matrix<OUTPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1> {
        for i in matrix.elements_mut() {
            *i *= usize::from(i > &mut 0.) as f64;
        }

        matrix
    }
}
