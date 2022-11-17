pub trait Optimizer {
    fn learning_rate(&self) -> f64;
}

pub struct StochasticGradientDescent {
    multiplier: f64,
}

impl StochasticGradientDescent {
    pub fn new(num_iterations: f64, learning_rate: f64) -> Self {
        Self {
            multiplier: learning_rate / num_iterations,
        }
    }
}

impl Optimizer for StochasticGradientDescent {
    fn learning_rate(&self) -> f64 {
        self.multiplier
    }
}
