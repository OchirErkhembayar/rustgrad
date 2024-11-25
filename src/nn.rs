use rand::Rng;

use crate::engine::Val;

/// Individual neurons that will take in several [`Val`]s as input and do elementwise
/// multiplication with them and it's own weights
pub struct Neuron {
    bias: Val,
    weights: Vec<Val>,
}

/// A layer of neurons. The [`MLP`] is constructed with several of these of varying sizes
pub struct Layer {
    neurons: Vec<Neuron>,
}

/// Multi layer perceptron model
///
/// Consists of several layers that the input will pass through
pub struct MLP {
    layers: Vec<Layer>,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        assert!(nin > 0);
        let mut rng = rand::thread_rng();
        let weights = Vec::from_iter((0..nin).map(|_| rng.gen_range(-1.0..=1.0).into()));
        Self {
            bias: rng.gen_range(-1.0..=1.0).into(),
            weights,
        }
    }

    pub fn call(&self, input: &[Val]) -> Val {
        assert_eq!(input.len(), self.weights.len());
        input
            .iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w * &self.bias)
            .reduce(|acc, v| acc + v)
            .map(|v| v.tanh())
            .unwrap()
    }

    pub fn parameters(&self) -> Vec<Val> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }

    pub fn zero_grad(&self) {
        self.weights.iter().for_each(|w| {
            w.zero_grad();
        });
        self.bias.zero_grad();
    }
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = Vec::from_iter((0..nout).map(|_| Neuron::new(nin)));
        Self { neurons }
    }

    pub fn call(&self, input: &[Val]) -> Vec<Val> {
        self.neurons.iter().map(|n| n.call(input)).collect()
    }

    pub fn parameters(&self) -> Vec<Val> {
        self.neurons.iter().fold(vec![], |mut acc, neuron| {
            acc.append(&mut neuron.parameters());
            acc
        })
    }

    pub fn zero_grad(&self) {
        self.neurons.iter().for_each(|n| n.zero_grad());
    }
}

impl MLP {
    /// Creates a new [`MLP`].
    pub fn new(layers: &[usize]) -> Self {
        let layers = layers.windows(2).map(|w| Layer::new(w[0], w[1])).collect();
        Self { layers }
    }

    /// Run the model with an input of values
    pub fn call(&self, input: &[Val]) -> Vec<Val> {
        let input = input.iter().map(|i| i.into()).collect::<Vec<_>>();
        self.layers
            .iter()
            .fold(input.to_owned(), |input, layer| layer.call(&input))
    }

    /// Collect all the weights of this model
    pub fn parameters(&self) -> Vec<Val> {
        self.layers.iter().fold(vec![], |mut acc, layer| {
            acc.append(&mut layer.parameters());
            acc
        })
    }

    /// Reset the gradients of all parameters inside this model
    pub fn zero_grad(&self) {
        self.layers.iter().for_each(|l| l.zero_grad());
    }
}
