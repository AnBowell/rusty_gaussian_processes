use futures::executor::block_on;

mod gaussian_processes;

use gaussian_processes::{multiple_gps, single_gp};

#[no_mangle]
pub extern "C" fn rust_multiple_gps(
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
    let future = multiple_gps(
        x_input_ptr,
        y_input_ptr,
        input_size,
        input_indices_ptr,
        input_indices_size,
        output_ptr,
        output_size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
    );
    block_on(future);
}

#[no_mangle]
pub extern "C" fn rust_single_gp(
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
    single_gp(
        x_input_ptr,
        y_input_ptr,
        input_size,
        output_ptr,
        output_size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
    );
}
