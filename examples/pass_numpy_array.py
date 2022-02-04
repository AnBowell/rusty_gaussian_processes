import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from ctypes import POINTER, c_uint32, c_size_t
from time import perf_counter

lib_path = "target/release/rusty_gaussian_processes.dll"  # in the windows version


lib = ctypes.cdll.LoadLibrary(lib_path)


input_x = np.arange(0, 1000000, 1, dtype=np.float64)

# input_x = np.full(two_d_shape, 1.0)


input_y = np.empty(1000000, dtype=np.float64)


output = np.empty(input_y.size, dtype=np.float64)

ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")


lib.call.argtypes = (ND_POINTER_2, ND_POINTER_2, c_size_t, ND_POINTER_1, c_size_t)

lib.call.restype = None


if not input_x.flags["C_CONTIGUOUS"]:
    print("Making array C_CONTIGUOUS")
    input_x = np.ascontiguousarray(input_x)


if not input_y.flags["C_CONTIGUOUS"]:
    print("Making array C_CONTIGUOUS")
    input_y = np.ascontiguousarray(input_y)

if not output.flags["C_CONTIGUOUS"]:
    print("Making array C_CONTIGUOUS")
    output = np.ascontiguousarray(output)


start_time = perf_counter()

result_count = lib.call(input_x, input_y, input_x.size, output, output.size)

end_time = perf_counter()

print("Time taken Rust: {}".format(end_time - start_time))


start_time = perf_counter()

x = np.power(input_x, 3)

end_time = perf_counter()
print("Time taken numpy: {}".format(end_time - start_time))
