use rand::rngs::ThreadRng;

use neural_net::{activation::ReLU, cost, LinearLayer, Matrix, Model};

#[derive(Default)]
struct SimplestNet {
    input: LinearLayer<1, 1, ReLU>,
    linear_1: LinearLayer<1, 1, ReLU>,
}

impl SimplestNet {
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

impl Model<1, 1> for SimplestNet {
    fn forward(&self, matrix: &Matrix<1, 1>) -> Matrix<1, 1> {
        let matrix = self.input.forward(matrix);
        self.linear_1.forward(&matrix)
    }
}

#[test]
fn new_empty() {
    let mut rng = rand::thread_rng();

    let network = SimplestNet::random_new(&mut rng);

    let output = network.forward(&Matrix::default());

    // assert_eq!(output.sum(), 0.);

    let cost = cost(
        network,
        Matrix::random_new(&mut rng),
        Matrix::from_arrays([[1.]]),
    );

    println!("{}", cost);
}

#[test]
fn training() {
    let mut network = SimplestNet::new();

    let matrix_0 = Matrix::from_arrays([[0.5]]);
    let matrix_1 = network.input.forward(&matrix_0);
    let output = network.linear_1.forward(&matrix_1);

    let cost = (output - Matrix::from_arrays([[1.]])).length_sqrd();

    let m_l_partial =
        (2. * output.elements().next().unwrap() - 2. * cost) * matrix_1.elements().next().unwrap();

    println!("{m_l_partial}");
}
