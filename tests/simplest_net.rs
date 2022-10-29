use rand::rngs::ThreadRng;

use neural_net::{cost, LinearLayer, Matrix, Model, Relu};

#[derive(Default, Debug)]
struct SimplestNet {
    input: Relu<1, 1, LinearLayer<1, 1>>,
    linear_1: Relu<1, 1, LinearLayer<1, 1>>,
    linear_2: Relu<1, 1, LinearLayer<1, 1>>,
    output: Relu<1, 1, LinearLayer<1, 1>>,
}

impl SimplestNet {
    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: Relu::new(LinearLayer::random_new(rng)),
            linear_1: Relu::new(LinearLayer::random_new(rng)),
            linear_2: Relu::new(LinearLayer::random_new(rng)),
            output: Relu::new(LinearLayer::random_new(rng)),
        }
    }
}

impl Model<1, 1> for SimplestNet {
    fn forward(&self, matrix: &Matrix<1, 1>) -> Matrix<1, 1> {
        let matrix = self.input.forward(matrix);
        let matrix = self.linear_1.forward(&matrix);
        let matrix = self.linear_2.forward(&matrix);
        self.output.forward(&matrix)
    }
}

#[test]
fn new_empty() {
    let mut rng = rand::thread_rng();

    let network = SimplestNet::random_new(&mut rng);

    let output = network.forward(&Matrix::default());

    // assert_eq!(output.sum(), 0.);

    let cost = cost(network, Matrix::random_new(&mut rng), Matrix::from_arrays([[1.]]));

    println!("{}", cost);
}
