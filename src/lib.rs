use futures::executor::block_on;

mod gaussian_processes;

use gaussian_processes::{single_gp, thread_gps};

#[no_mangle]
pub extern "C" fn rust_thread_gps(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    output_ptr: *mut f64,
    output_size: usize,
    length_scale: f64,
    amplitude: f64,
) {
    let future = thread_gps(
        x_input_ptr,
        y_input_ptr,
        input_size,
        output_ptr,
        output_size,
        length_scale,
        amplitude,
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
