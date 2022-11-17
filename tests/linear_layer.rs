use neural_net::{
    activation::ReLU,
    cost::{LossFunction, MSEloss},
    Backpropagation, Differentiable, LinearLayer, Matrix, Model,
};

#[cfg(feature = "random_generation")]
use rand::rngs::ThreadRng;

// #[derive(Default)]
struct TestNetwork {
    input: LinearLayer<2, 3, ReLU>,
    output: LinearLayer<3, 2, ReLU>,
}

impl TestNetwork {
    pub fn new() -> Self {
        Self {
            input: LinearLayer::new(
                Matrix::from_arrays([
                    [0.25, 0.2, 0.666666666667],
                    [0.833333333333, 0.75, 0.444444444444],
                ]),
                Matrix::from_arrays([[0.125, 0.25, 0.166666666667]]),
            ),
            output: LinearLayer::new(
                Matrix::from_arrays([[1.2, 1.75], [0.625, 1.25], [0.5, 0.75]]),
                Matrix::from_arrays([[0.666666666667, 0.666666666667]]),
            ),
        }
    }

    pub fn random_new(rng: &mut ThreadRng) -> Self {
        Self {
            input: LinearLayer::random_new(rng),
            output: LinearLayer::random_new(rng),
        }
    }
}

impl Model<2, 2> for TestNetwork {
    fn forward(&self, matrix: &Matrix<2, 1>) -> Matrix<2, 1> {
        let matrix = self.input.forward(matrix);
        self.output.forward(&matrix)
    }
}

impl Differentiable<2, 2> for TestNetwork {
    fn calculate_grads(
        &mut self,
        backpropagation: Backpropagation<2>,
        matrix: Matrix<2, 1>,
    ) -> Backpropagation<2> {
        let matrix_0 = self.input.forward(&matrix);
        let matrix_1 = self.output.forward(&matrix_0);

        let mut loss = MSEloss {
            expected: Matrix::from_arrays([[1.5, 2.]]),
        };

        // holds derivatives to the output neurons
        let backpropagation = loss.calculate_grads(backpropagation, matrix_1);

        assert_eq!(
            backpropagation.total_derivatives,
            Matrix::from_arrays([[2.8510416666664, 4.604166666666125]])
        );

        let backpropagation = self.output.calculate_grads(backpropagation, matrix_0);

        let backpropagation = self.input.calculate_grads(backpropagation, matrix);

        backpropagation
    }
}

#[test]
fn new_empty() {
    let mut network = TestNetwork::new();

    let backpropagation = Backpropagation::default();
    network.calculate_grads(backpropagation, Matrix::from_arrays([[1., 1.5]]));

    let loss = MSEloss {
        expected: Matrix::from_arrays([[1.5, 2.]]),
    };

    let output = network.forward(&Matrix::from_arrays([[1., 1.5]]));

    let output = loss.loss_function(output);

    assert_eq!(output, 14.66339463975369);
}

#[cfg(feature = "random_generation")]
#[test]
fn new_random() {
    let mut rng = rand::thread_rng();

    let network = TestNetwork::random_new(&mut rng);

    // println!("{:?}", network);

    let output = network.forward(&Matrix::random_new(&mut rng));

    println!("{:?}", output);
}
