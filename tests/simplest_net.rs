use rand::rngs::ThreadRng;

use neural_net::{activation::ReLU, cost, GhostToken, LinearLayer, Matrix, Model};

#[derive(Default)]
struct SimplestNet<'id> {
    input: LinearLayer<'id, 1, 1, ReLU>,
    linear_1: LinearLayer<'id, 1, 1, ReLU>,
}

impl<'id> SimplestNet<'id> {
    pub fn new() -> Self {
        let input = LinearLayer::new(Matrix::from_arrays([[1.]]), Matrix::from_arrays([[0.]]));

        let linear_1 = LinearLayer::new(Matrix::from_arrays([[1.5]]), Matrix::from_arrays([[0.]]));

        Self { input, linear_1 }
    }
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: LinearLayer::random_new(rng),
            linear_1: LinearLayer::random_new(rng),
        }
    }
}

impl<'id> Model<'id, 1, 1> for SimplestNet<'id> {
    fn forward(&self, token: &GhostToken<'id>, matrix: &Matrix<1, 1>) -> Matrix<1, 1> {
        let matrix = self.input.forward(token, matrix);
        self.linear_1.forward(token, &matrix)
    }
}

#[test]
fn new_empty() {
    GhostToken::new(|token| {
        let mut rng = rand::thread_rng();

        let network = SimplestNet::random_new(&mut rng);

        let output = network.forward(&token, &Matrix::default());

        // assert_eq!(output.sum(), 0.);

        let cost = cost(
            network,
            &token,
            Matrix::random_new(&mut rng),
            Matrix::from_arrays([[1.]]),
        );

        println!("{}", cost);
    });
}

#[test]
fn training() {
    GhostToken::new(|token| {
        let mut network = SimplestNet::new();

        let matrix_0 = Matrix::from_arrays([[0.5]]);
        let matrix_1 = network.input.forward(&token, &matrix_0);
        let output = network.linear_1.forward(&token, &matrix_1);

        let cost = (output - Matrix::from_arrays([[1.]])).length_sqrd();

        let m_l_partial = (2. * output.elements().next().unwrap() - 2. * cost)
            * matrix_1.elements().next().unwrap();

        println!("{m_l_partial}");
    })
}
