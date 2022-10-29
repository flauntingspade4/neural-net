use neural_net::{LinearLayer, Matrix, Model, Relu};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

#[derive(Default, Debug)]
struct TestNetwork {
    input: Relu<3, 5, LinearLayer<3, 5>>,
    linear_1: Relu<5, 64, LinearLayer<5, 64>>,
    output: Relu<64, 10, LinearLayer<64, 10>>,
}

impl TestNetwork {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: Relu::new(LinearLayer::random_new(rng)),
            linear_1: Relu::new(LinearLayer::random_new(rng)),
            output: Relu::new(LinearLayer::random_new(rng)),
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
