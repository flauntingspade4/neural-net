use crate::{ghost::GhostToken, Matrix, Model};

pub fn cost<'id, const INPUT_LEN: usize, const OUTPUT_LEN: usize>(
    model: impl Model<'id, INPUT_LEN, OUTPUT_LEN>,
    token: &GhostToken<'id>,
    input: Matrix<INPUT_LEN, 1>,
    desired_output: Matrix<OUTPUT_LEN, 1>,
) -> f64 {
    (model.forward(token, &input) - desired_output).length_sqrd()
}
