import numpy as np
from numba import cuda
import math


def euclidius(a, b):
    while b:
        a, b = b, a % b
    return a


@cuda.jit
def gcd(ar, result):
    arlen = ar.shape[1]
    # ar_local_mem = cuda.shared.array(ar.shape[0], dtype=int)
    row, col = cuda.grid(2)
    # if pos < arlen:
    #     ar_local_mem[pos] = ar[pos]
    # cuda.syncthreads()
    if row < ar.shape[0]:
        while arlen > 1:
            if col < (arlen//2):
                while ar[row, col * 2 + 1]:
                    ar[row, col * 2], ar[row, col * 2 + 1] = ar[row, col*2+1], ar[row, col * 2] % ar[row, col * 2 + 1]
                ar[row, col] = ar[row, col*2]
                #ar[pos] = euclidius(ar[pos*2], ar[pos*2+1])
            if math.ceil(arlen/2) > arlen//2:
                ar[row, int(arlen//2)] = ar[row, int(arlen-1)]
            arlen = math.ceil(arlen/2)
            cuda.syncthreads()
        result[row] = ar[row, 0]


test_array = np.array([[22, 100, 8, 4, 16, 10, 6],[22, 100, 8, 4, 16, 10, 6],[22, 100, 8, 4, 16, 10, 6],], dtype=int)
# gcds_global_mem = cuda.device_array(test_array.shape[0], dtype=int)
# threadsperblock = (16, 16)
# blockspergrid_x = int(math.ceil(test_array.shape[0] / threadsperblock[0]))
# blockspergrid_y = int(math.ceil(test_array.shape[1] /threadsperblock[1]))
# blockspergrid = (blockspergrid_x, blockspergrid_y)
# gcd[blockspergrid, threadsperblock](test_array, gcds_global_mem)
# print(gcds_global_mem.copy_to_host())
for i in range((test_array.shape[0] // 2)*2, 3):
    print(i)