use rusty_machine::learning::gp::{ConstMean, GaussianProcess};
use rusty_machine::learning::{toolkit::kernel, SupModel};
use rusty_machine::linalg::{Matrix, Vector};

use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

pub async fn multiple_gps(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    input_indices_ptr: *mut usize,
    input_indices_size: usize,
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

    let input_indices: &mut [usize] = unsafe {
        assert!(!input_indices_ptr.is_null());
        std::slice::from_raw_parts_mut(input_indices_ptr, input_indices_size)
    };

    let output: &mut [f64] = unsafe {
        assert!(!output_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, output_size)
    };

    let ker = kernel::SquaredExp::new(length_scale, amplitude);

    let zero_mean = ConstMean::default();

    let mut handles: Vec<JoinHandle<Vector<f64>>> = Vec::with_capacity(input_indices_size);

    for i in 0..input_indices_size {
        let (x_input_slice, y_input_slice) = if i + 1_usize >= input_indices_size {
            (&x_input[input_indices[i]..], &y_input[input_indices[i]..])
        } else {
            (
                &x_input[input_indices[i]..input_indices[i + 1]],
                &y_input[input_indices[i]..input_indices[i + 1]],
            )
        };

        handles.push(rt.spawn(async move {
            let mut x_input_vector = x_input_slice.to_vec();

            let training_x = Matrix::new(x_input_slice.len(), 1, x_input_slice);

            let training_y = Vector::new(y_input_slice);
            // Has to be created in the thread - no clone trait.
            let mut gp = GaussianProcess::new(ker, zero_mean, noise);

            gp.train(&training_x, &training_y).unwrap();

            let final_value = x_input_vector.last().unwrap();

            let mut forecast_days: Vec<f64> = (1..forecast_amount + 1_i64)
                .map(|i| ((i * forecast_spacing) as f64) + final_value)
                .collect();

            x_input_vector.append(&mut forecast_days);

            let smoothed_and_forecast_x = Matrix::new(x_input_vector.len(), 1, x_input_vector);

            return gp.predict(&smoothed_and_forecast_x).unwrap();
        }));
    }

    // TODO There will probably be a more efficient way of doing this.
    for i in 0..input_indices_size {
        let result = (&mut handles[i]).await.unwrap().into_vec();

        let this_index = input_indices[i] + (i * forecast_amount as usize);

        for result_index in 0..result.len() {
            output[this_index + result_index] = result[result_index];
        }
    }
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
