use neural_net::{activation::ReLU, LinearLayer, Matrix, Model};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

#[derive(Default, Debug)]
struct TestNetwork {
    input: LinearLayer<3, 5, ReLU>,
    linear_1: LinearLayer<5, 64, ReLU>,
    output: LinearLayer<64, 10, ReLU>,
}

impl TestNetwork {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: LinearLayer::random_new(rng),
            linear_1: LinearLayer::random_new(rng),
            output: LinearLayer::random_new(rng),
        }
    }
}

impl Model<3, 10> for TestNetwork {
    fn forward(&self, matrix: &Matrix<3, 1>) -> Matrix<10, 1> {
        let matrix = self.input.forward(matrix);
        let matrix = self.linear_1.forward(&matrix);
        self.output.forward(&matrix)
    }
}

#[test]
fn new_empty() {
    let network = TestNetwork::default();

    let output = network.forward(&Matrix::default());

    assert_eq!(output.elements().sum::<f64>(), 0.);
}

#[cfg(feature = "random_generation")]
#[test]
fn new_random() {
    let mut rng = rand::thread_rng();

    let network = TestNetwork::random_new(&mut rng);

    println!("{:?}", network);

    let output = network.forward(&Matrix::random_new(&mut rng));

    println!("{:?}", output);
}
