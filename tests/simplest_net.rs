use rand::rngs::ThreadRng;

use neural_net::{cost, LinearLayer, Model, Relu, Tensor};

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
    fn forward(&self, tensor: &Tensor<1>) -> Tensor<1> {
        let tensor = self.input.forward(tensor);
        let tensor = self.linear_1.forward(&tensor);
        let tensor = self.linear_2.forward(&tensor);
        self.output.forward(&tensor)
    }
}

#[test]
fn new_empty() {
    let mut rng = rand::thread_rng();

    let network = SimplestNet::random_new(&mut rng);

    let output = network.forward(&Tensor::default());

    // assert_eq!(output.sum(), 0.);

    let cost = cost(network, Tensor::random_new(&mut rng), Tensor::new([1.]));

    println!("{}", cost);
}
