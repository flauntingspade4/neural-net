use crate::{Matrix, Model};

#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
pub struct Relu<const INPUT_LEN: usize, const OUTPUT_LEN: usize, M: Model<INPUT_LEN, OUTPUT_LEN>> {
    model: M,
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, M: Model<INPUT_LEN, OUTPUT_LEN>>
    Relu<INPUT_LEN, OUTPUT_LEN, M>
{
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

impl<const INPUT_LEN: usize, const OUTPUT_LEN: usize, M: Model<INPUT_LEN, OUTPUT_LEN>>
    Model<INPUT_LEN, OUTPUT_LEN> for Relu<INPUT_LEN, OUTPUT_LEN, M>
{
    fn forward(&self, matrix: &Matrix<INPUT_LEN, 1>) -> Matrix<OUTPUT_LEN, 1> {
        let mut matrix = self.model.forward(matrix);

        for i in matrix.elements_mut() {
            *i = (i > &mut 0.) as usize as f64 * *i;
        }

        matrix
    }
}
