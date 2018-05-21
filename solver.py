from functools import reduce
from numba import cuda, int32
import numpy as np
import math


def read_file(input_name):
    input_arr = []
    with open(input_name) as input_file:
        for line in input_file:
            l = line.split()
            int_l = []
            for ch in l:
                int_l.append(int(ch))
            input_arr.append(int_l)
    return input_arr

# TPB=8
# @cuda.jit
# def fast_matmul(A, B, C):
#     """
#     Perform matrix multiplication of C = A * B
#     Each thread computes one element of the result matrix C
#     """
#
#     # Define an array in the shared memory
#     # The size and type of the arrays must be known at compile time
#     sA = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
#     sB = cuda.shared.array(shape=(TPB, TPB), dtype=int32)
#
#     x, y = cuda.grid(2)
#
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#
#     if x >= C.shape[0] or y >= C.shape[1]:
#         # Quit if (x, y) is outside of valid C boundary
#         return
#
#     # Each thread computes one element in the result matrix.
#     # The dot product is chunked into dot products of TPB-long vectors.
#     tmp = 0
#     for i in range(int(math.ceil(A.shape[1] / TPB))):
#         # Preload data into shared memory
#         if ty + i*TPB < A.shape[1]:
#             sA[tx, ty] = A[x, ty + i * TPB]
#         if tx + i * TPB < B.shape[0]:
#             sB[tx, ty] = B[tx + i * TPB, y]
#
#         # Wait until all threads finish preloading
#         cuda.syncthreads()
#
#         # Computes partial product on the shared memory
#         for j in range(TPB):
#             if (j + i*TPB) < A.shape[1]:
#                 tmp += sA[tx, j] * sB[j, ty]
#
#         # Wait until all threads finish computing
#         cuda.syncthreads()
#
#     # for i in range((int(A.shape[1]//TPB))*TPB, A.shape[1]):
#     #     tmp += A[x, i] * B[i, y]
#     #     cuda.syncthreads()
#
#     C[x, y] = tmp


@cuda.jit
def create_pre_basis(equation, pre_basis):
    i_not_zero = 0
    for i in range(0, equation.shape[0]):  # do it outside of this cuda function
        if equation[i] != 0:
            i_not_zero = i
            break
    pos = cuda.grid(1)
    if pos < pre_basis.shape[0]:
        for p in range(pre_basis[pos].shape[0]):
            pre_basis[pos][p] = 0
        if pos >= i_not_zero:
            pre_basis[pos][i_not_zero] = equation[pos+1] * -1
            pre_basis[pos][pos+1] = equation[i_not_zero]
        else:
            pre_basis[pos][i_not_zero] = equation[pos] * -1
            pre_basis[pos][pos] = equation[i_not_zero]


def substitute(equation, pre_basis):
    result = []
    for vector in pre_basis:
        one_vector_result = 0
        for c in range(0, len(equation)):
            one_vector_result += vector[c]*equation[c]
        result.append(one_vector_result)
    return result


@cuda.jit
def cuda_substitute(equation, pre_basis, result):
    pos = cuda.grid(1)
    if pos < result.shape[0]:
        one_vector_result = 0
        for c in range(0, equation.shape[0]):
            one_vector_result += pre_basis[pos, c] * equation[c]
        result[pos] = one_vector_result

@cuda.jit
def multiply_pre_basis(small_pre_basis, big_pre_basis, result):
    """
    multiplies pre basis from equation gained from substitution and pre basis from initial equation
    :param big_pre_basis:
    :param small_pre_basis:
    :return: result
    """
    row, col = cuda.grid(2)
    if row < result.shape[0] and col < result.shape[1]:
        tmp = 0
        for k in range(small_pre_basis.shape[1]):
            tmp += small_pre_basis[row, k] * big_pre_basis[k, col]
        result[row, col] = tmp

    # for small_vector in small_pre_basis:
    #     one_vector_result = [0] * len(big_pre_basis[0])
    #     for i_small_vector in range(0, len(small_vector)):
    #         if small_vector[i_small_vector] != 0:
    #             for c in range(len(big_pre_basis[0])):
    #                 one_vector_result[c] += big_pre_basis[i_small_vector][c] * small_vector[i_small_vector]  # possible race condition
    #     result.append(one_vector_result)
    # return result


# cudareduced_gcd = cuda.reduce(math.gcd)
# @cuda.reduce
# def reduced_cuda_GCD(a, b):  # Greatest Common Divisor
#     while b:
#         a, b = b, a % b
#     return a

# @cuda.jit
# def find_gcds(ar, gcds):
#     pos = cuda.grid(1)
#     if pos < gcds.shape[0]:
#         cudareduced_gcd(ar[pos], res=gcds)
        # gcds[pos] = cuda.reduce(math.gcd, ar[pos])


@cuda.jit
def cuda_gcd(ar, work_copy):
    arlen = work_copy.shape[1]
    # ar_local_mem = cuda.shared.array(ar.shape[0], dtype=int)
    row, col = cuda.grid(2)
    # if pos < arlen:
    #     ar_local_mem[pos] = ar[pos]
    if row < ar.shape[0] and col < ar.shape[1]:
        work_copy[row, col] = ar[row, col]
        cuda.syncthreads()
        while arlen > 1:
            if col < (arlen//2):
                while work_copy[row, col * 2 + 1]:
                    work_copy[row, col * 2], work_copy[row, col * 2 + 1] = work_copy[row, col * 2 + 1], work_copy[row, col * 2] % work_copy[row, col * 2 + 1]
                work_copy[row, col] = work_copy[row, col * 2]
                #ar[pos] = euclidius(ar[pos*2], ar[pos*2+1])
            if math.ceil(arlen/2) > arlen//2:
                work_copy[row, int(arlen // 2)] = work_copy[row, int(arlen - 1)]
            arlen = math.ceil(arlen/2)
            cuda.syncthreads()
        # d = work_copy[row][0]
        # if d != 1 and d != 0:
        #     ar[row, col] = ar[row, col] // d


def find_gcds(ar):
    gcds = np.empty(ar.shape[0], dtype=int)
    for i in range(ar.shape[0]):
        gcds[i] = reduce(math.gcd, ar[i])
    return gcds


@cuda.jit
def simplify(ar, gcds):
    row, col = cuda.grid(2)
    if row < ar.shape[0] and col < ar.shape[1]:
        d = gcds[row][0]
        if d != 1 and d != 0:
            ar[row, col] = ar[row, col] // d


def solv(input_arr):
    # pre_basis_main = create_pre_basis(input_arr[0])
    pre_basis_main = cuda.device_array(((len(input_arr[0])-1), len(input_arr[0])), dtype=int)

    threadsperblock = 512
    blockspergrid = math.ceil((len(input_arr[0])-1)/threadsperblock)

    create_pre_basis[blockspergrid, threadsperblock](np.array(input_arr[0], dtype=int), pre_basis_main)
    for Li in range(1, len(input_arr)):
        subresult_global_mem = cuda.device_array(pre_basis_main.shape[0], dtype=int)
        next_equation = np.array(input_arr[Li], dtype=int)
        threadsperblock = 256
        blockspergrid = math.ceil(pre_basis_main.shape[0]/threadsperblock)
        cuda_substitute[blockspergrid, threadsperblock](next_equation, pre_basis_main, subresult_global_mem)
        #debug_subresult_global_mem = subresult_global_mem.copy_to_host()
        # Y = subresult_global_mem.copy_to_host().tolist()

        #pre_basis_Y = create_pre_basis(Y)
        pre_basis_Y = cuda.device_array((subresult_global_mem.shape[0]-1, subresult_global_mem.shape[0]), dtype=int)
        threadsperblock = 512
        blockspergrid = math.ceil((subresult_global_mem.shape[0] - 1) / threadsperblock)
        create_pre_basis[blockspergrid, threadsperblock](subresult_global_mem, pre_basis_Y)

        #debug_pre_basis_Y = pre_basis_Y.copy_to_host()
        #mult_pre_basis_Y = np.array(pre_basis_Y, dtype=int)
        mult_result_global_mem = cuda.device_array((pre_basis_Y.shape[0], pre_basis_main.shape[1]), dtype=int)

        threadsperblock = (8, 8)
        blockspergrid_x = int(math.ceil(pre_basis_Y.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(pre_basis_main.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        multiply_pre_basis[blockspergrid, threadsperblock](pre_basis_Y, pre_basis_main, mult_result_global_mem)
        # fast_matmul[blockspergrid, threadsperblock](pre_basis_Y, pre_basis_main, mult_result_global_mem)


        pre_basis_main = mult_result_global_mem

        gcds_global_mem = cuda.device_array(pre_basis_main.shape, dtype=int)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(pre_basis_main.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(pre_basis_main.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        cuda_gcd[blockspergrid, threadsperblock](pre_basis_main, gcds_global_mem)
        # gcds = find_gcds(pre_basis_main.copy_to_host())
        # gcds_global_mem = cuda.to_device(gcds)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(pre_basis_main.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(pre_basis_main.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #simplify[blockspergrid, threadsperblock](pre_basis_main, gcds_global_mem)
    return pre_basis_main.copy_to_host()


def control_main(input_name):
    return solv(read_file(input_name))

def timetest():
    solv(read_file(file))

if __name__ == "__main__":
    file = 'input40x61(10x30).txt'
    print(solv(read_file(file)))
    import timeit
    print(timeit.timeit("timetest()", setup="from __main__ import timetest", number=100))
