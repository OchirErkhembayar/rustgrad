use engine::Val;
use num_traits::pow::Pow;

pub mod engine;
pub mod nn;

pub fn loss(pred: &[Val], actual: &[Val]) -> Val {
    pred.iter()
        .zip(actual)
        .map(|(pred, actual)| (pred - actual).pow(2.0))
        .reduce(|acc, curr| acc + curr)
        .unwrap()
}
