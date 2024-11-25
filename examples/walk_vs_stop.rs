use rustgrad::{engine::Val, nn::MLP};

use num_traits::pow::Pow;

const ALPHA: f32 = 0.1;

fn main() {
    let mlp = MLP::new(&[3, 5, 5, 1]);
    gradient_descent(&mlp, 20);
}

fn gradient_descent(mlp: &MLP, epochs: usize) {
    let xs = vec![
        // [Red, Yellow, Green] lights
        // 0 = Completely dark
        // 1 = Fully bright
        vec![Val::new(1.0), Val::new(0.0), Val::new(0.0)],
        vec![Val::new(0.25), Val::new(1.0), Val::new(0.1)],
        vec![Val::new(0.1), Val::new(0.25), Val::new(0.95)],
        vec![Val::new(0.1), Val::new(0.3), Val::new(0.85)],
    ];
    // 1 = Go
    // 0 = Cannot go
    let ys = vec![0.0, 0.0, 1.0, 1.0];
    for i in 0..epochs {
        println!("Iteration: {}", i + 1);
        // Forward pass
        let ypred: Vec<Val> = xs.iter().map(|input| mlp.call(input)).flatten().collect();
        println!(
            "Pred: {:?}\nActual: {:?}",
            ypred.iter().map(|v| v.data()).collect::<Vec<_>>(),
            ys
        );

        // Loss function
        let loss = ypred
            .iter()
            .zip(&ys)
            .map(|(pred, actual)| (pred.clone() - *actual).pow(2.0))
            .reduce(|acc, curr| acc + curr)
            .unwrap();
        println!("Loss: {}", loss.data());

        // Resetting the gradients
        mlp.zero_grad();

        // Calculating the gradients
        loss.backward();

        // Updating the weights
        mlp.parameters().iter_mut().for_each(|p| {
            p.add_data(-ALPHA * p.grad());
        });
        println!();
    }
}
