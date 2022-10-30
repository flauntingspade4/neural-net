use std::marker::PhantomData;

use crate::{ghost::GhostToken, Matrix, Model};

#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
pub struct Relu<
    'id,
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
    M: Model<'id, INPUT_LEN, OUTPUT_LEN>,
> {
    model: M,
    _phantom: PhantomData<&'id M>,
}

impl<
        'id,
        const INPUT_LEN: usize,
        const OUTPUT_LEN: usize,
        M: Model<'id, INPUT_LEN, OUTPUT_LEN>,
    > Relu<'id, INPUT_LEN, OUTPUT_LEN, M>
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            _phantom: PhantomData,
        }
    }
}

impl<
        'id,
        const INPUT_LEN: usize,
        const OUTPUT_LEN: usize,
        M: Model<'id, INPUT_LEN, OUTPUT_LEN>,
    > Model<'id, INPUT_LEN, OUTPUT_LEN> for Relu<'id, INPUT_LEN, OUTPUT_LEN, M>
{
    fn forward(
        &self,
        token: &GhostToken<'id>,
        matrix: &Matrix<INPUT_LEN, 1>,
    ) -> Matrix<OUTPUT_LEN, 1> {
        let mut matrix = self.model.forward(token, matrix);

        for i in matrix.elements_mut() {
            *i = (i > &mut 0.) as usize as f64 * *i;
        }

        matrix
    }
}
