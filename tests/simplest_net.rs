use rand::rngs::ThreadRng;

use neural_net::{
    activation::ReLU, cost::LossFunction, cost::MSEloss, optim::Optimizer,
    optim::StochasticGradientDescent, Backpropagation, Differentiable, LinearLayer, Matrix, Model,
};

static LEARNING_RATE: f64 = 1.;

#[derive(Default, Debug)]
struct SimplestNet {
    input: LinearLayer<2, 5, ReLU>,
    hidden: LinearLayer<5, 3, ReLU>,
    output: LinearLayer<3, 1, ReLU>,
}

impl SimplestNet {
    fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "random_generation")]
    fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: LinearLayer::random_new(rng),
            hidden: LinearLayer::random_new(rng),
            output: LinearLayer::random_new(rng),
        }
    }

    fn calculate_grads(
        &mut self,
        backpropagation: Backpropagation<1>,
        input: Matrix<2, 1>,
        expected: Matrix<1, 1>,
    ) {
        let matrix_0 = self.input.forward(&input);
        let matrix_1 = self.hidden.forward(&matrix_0);
        let matrix_2 = self.output.forward(&matrix_1);

        let mut loss = MSEloss { expected };

        // Initialize total derivatives
        let backpropagation = loss.calculate_grads(backpropagation, matrix_2);

        let backpropagation = self.output.calculate_grads(backpropagation, matrix_1);
        let backpropagation = self.hidden.calculate_grads(backpropagation, matrix_0);
        self.input.calculate_grads(backpropagation, input);
    }

    pub fn apply_gradients(&mut self, optim: &impl Optimizer) {
        self.input.apply_gradients(optim);
        self.hidden.apply_gradients(optim);
        self.output.apply_gradients(optim);
        self.input.zero_grad();
        self.hidden.zero_grad();
        self.output.zero_grad();
    }
}

impl Model<2, 1> for SimplestNet {
    fn forward(&self, matrix: &Matrix<2, 1>) -> Matrix<1, 1> {
        let matrix = self.input.forward(matrix);
        let matrix = self.hidden.forward(&matrix);
        self.output.forward(&matrix)
    }
}

#[test]
fn new_empty() {
    let mut rng = rand::thread_rng();

    let network = SimplestNet::random_new(&mut rng);

    let output = network.forward(&Matrix::default());

    // assert_eq!(output.sum(), 0.);

    /*let cost = cost(
        network,
        Matrix::random_new(&mut rng),
        Matrix::from_arrays([[1.]]),
    );

    println!("{}", cost);*/
}

#[test]
#[cfg(feature = "random_generation")]
fn training() {
    use std::time::Instant;

    use rand::Rng;

    let mut rng = rand::thread_rng();

    let mut network = SimplestNet::random_new(&mut rng);

    let start = Instant::now();

    for _ in 0..500 {
        for _ in 0..5000 {
            let lhs: bool = rng.gen();
            let rhs: bool = rng.gen();

            let expected = lhs & rhs;

            network.calculate_grads(
                Backpropagation::default(),
                Matrix::from_arrays([[usize::from(lhs) as f64, usize::from(rhs) as f64]]),
                Matrix::from_arrays([[usize::from(expected) as f64]]),
            );
        }

        let gradient_descent = StochasticGradientDescent::new(5000., LEARNING_RATE);

        network.apply_gradients(&gradient_descent);
    }

    println!("Took {}ms", start.elapsed().as_millis());

    println!("{:?}", network);

    let mut total_loss = 0.;

    for _ in 0..5000 {
        let lhs: bool = rng.gen();
        let rhs: bool = rng.gen();

        let expected = lhs & rhs;

        let output = network.forward(&Matrix::from_arrays([[
            usize::from(lhs) as f64,
            usize::from(rhs) as f64,
        ]]));

        let loss = MSEloss {
            expected: Matrix::from_arrays([[usize::from(expected) as f64]]),
        };

        total_loss += loss.loss_function(output);
    }

    total_loss *= 100. / 5000.;

    println!("{}% loss", total_loss);
}
