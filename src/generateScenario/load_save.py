import struct
import numpy as np
import time
import sys
import resource
start = time.time()
print("Translating bin to npy")
feasible = True

with open("./data_simulation.bin", "rb") as file:
    dim1 = struct.unpack("Q", file.read(8))[0]  # Read 64-bit unsigned integer (size_t)
    data = []
    for _ in range(dim1):
        dim2 = struct.unpack("Q", file.read(8))[0]
        vec2d = []
        for _ in range(dim2):
            dim3 = struct.unpack("Q", file.read(8))[0]
            vec1d = struct.unpack("{}i".format(dim3), file.read(dim3 * 4))  # Read int values
            vec2d.append(np.array(vec1d, dtype='b'))
        data.append(vec2d)
C = np.array(data, dtype='b')
print("Array size:", sys.getsizeof(C))

if feasible:
    C = C[:, :-1, :]  # Note this is to remove the always feasible satellite, this is

np.save('simulation_coverage_info.npy', C)

max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Maximum memory usage: {max_mem} bytes")  # 或者是 "KB" 视具体情况而定


print(f"Translation costs {time.time() - start}")
