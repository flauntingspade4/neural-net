use neural_net::{LinearLayer, Model, Relu, Tensor};

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
    fn forward(&self, tensor: &Tensor<3>) -> Tensor<10> {
        let tensor = self.input.forward(tensor);
        let tensor = self.linear_1.forward(&tensor);
        self.output.forward(&tensor)
    }
}

#[test]
fn new_empty() {
    let network = TestNetwork::default();

    let output = network.forward(&Tensor::default());

    assert_eq!(output.sum(), 0.);
}

#[cfg(feature = "random_generation")]
#[test]
fn new_random() {
    let mut rng = rand::thread_rng();

    let network = TestNetwork::random_new(&mut rng);

    println!("{:?}", network);

    let output = network.forward(&Tensor::random_new(&mut rng));

    println!("{:?}", output);
}
