use rand::Rng;

use crate::engine::Val;

pub struct Neuron {
    pub bias: Val,
    pub weights: Vec<Val>,
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

pub struct MLP {
    pub layers: Vec<Layer>,
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
            .map(|(x, w)| x.clone() * w.clone() * self.bias.clone())
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
    pub fn new(layers: &[usize]) -> Self {
        let layers = layers.windows(2).map(|w| Layer::new(w[0], w[1])).collect();
        Self { layers }
    }

    pub fn call(&self, input: &[Val]) -> Vec<Val> {
        let input = input.iter().map(|i| i.into()).collect::<Vec<_>>();
        self.layers
            .iter()
            .fold(input.to_owned(), |input, layer| layer.call(&input))
    }

    pub fn parameters(&self) -> Vec<Val> {
        self.layers.iter().fold(vec![], |mut acc, layer| {
            acc.append(&mut layer.parameters());
            acc
        })
    }

    pub fn zero_grad(&self) {
        self.layers.iter().for_each(|l| l.zero_grad());
    }
}
