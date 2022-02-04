use rusty_machine::learning::gp::{ConstMean, GaussianProcess};
use rusty_machine::learning::{toolkit::kernel, SupModel};
use rusty_machine::linalg::{Matrix, Vector};

use tokio::runtime::Runtime;
pub async fn thread_gps(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    output_ptr: *mut f64,
    output_size: usize,
    length_scale: f64,
    amplitude: f64,
) {
    let rt = Runtime::new().expect("Could not make new runtime.");

    let x_input: &mut [f64] = unsafe {
        assert!(!x_input_ptr.is_null());
        std::slice::from_raw_parts_mut(x_input_ptr, input_size)
    };

    let y_input: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(y_input_ptr, input_size)
    };

    let output: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, output_size)
    };

    // Store the output.
    let mut handles = Vec::with_capacity(input_size);

    (0..input_size).for_each(|i| {
        let single_input = x_input[i];
        handles.push(rt.spawn(async move { single_input.powf(3.) }))
    });
    // for i in 0..input_size {
    //     let single_input = x_input[i];

    //     handles.push(rt.spawn(async move { single_input.powf(3.) }));
    // }
    for (pos, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap();
        output[pos] = result;
    }

    // println!("Data: {:?}", x_input);
}

pub fn single_gp(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    output_ptr: *mut f64,
    output_size: usize,
    forecast_spacing: i64,
    forecast_amount: i64,
    length_scale: f64,
    amplitude: f64,
    noise: f64,
) {
    let rt = Runtime::new().expect("Could not make new runtime.");

    let x_input: &mut [f64] = unsafe {
        assert!(!x_input_ptr.is_null());
        std::slice::from_raw_parts_mut(x_input_ptr, input_size)
    };

    let y_input: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(y_input_ptr, input_size)
    };

    let output: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, output_size)
    };

    let mut x_input_vector = x_input.to_vec();

    let training_x = Matrix::new(input_size, 1, x_input);

    let training_y = Vector::new(y_input);

    let ker = kernel::SquaredExp::new(length_scale, amplitude);

    let zero_mean = ConstMean::default();

    let mut gp = GaussianProcess::new(ker, zero_mean, noise);

    gp.train(&training_x, &training_y).unwrap();

    let final_value = x_input_vector.last().unwrap();

    let mut forecast_days: Vec<f64> = (1..forecast_amount + 1_i64)
        .map(|i| ((i * forecast_spacing) as f64) + final_value)
        .collect();

    x_input_vector.append(&mut forecast_days);

    let smoothed_and_forecast_x = Matrix::new(x_input_vector.len(), 1, x_input_vector);

    let smoothed_data = gp.predict(&smoothed_and_forecast_x).unwrap();

    for i in 0..output_size {
        output[i] = smoothed_data[i]
    }
}
