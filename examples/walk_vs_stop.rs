use rustgrad::{engine::Val, loss, nn::MLP};

const ALPHA: f32 = 0.035;

fn main() {
    // Input layer: 3 inputs
    // 2 hidden layers of 5
    // A single output layer
    let mlp = MLP::new(&[3, 5, 5, 1]);
    gradient_descent(&mlp, 100);
}

/// The input is a list of sets of traffic lights ordered Red Yellow Green
/// The numeric values of the lights indicates the brightness
///
/// The model should make a prediction on whether or not it can go based on an
/// input of a set of lights
///
/// 0.0 -> 1.0 ranges from definitely not moving to definitely should move
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
    let ys = vec![Val::new(0.0), Val::new(0.0), Val::new(1.0), Val::new(1.0)];
    for i in 0..epochs {
        println!("Iteration: {}", i + 1);
        // Forward pass
        let ypred: Vec<Val> = xs.iter().map(|input| mlp.call(input)).flatten().collect();
        println!(
            "Pred: {:?}\nActual: {:?}",
            ypred.iter().map(|v| v.data()).collect::<Vec<_>>(),
            ys.iter().map(|v| v.data()).collect::<Vec<_>>(),
        );

        // Loss function
        let loss = loss(&ypred, &ys);
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
