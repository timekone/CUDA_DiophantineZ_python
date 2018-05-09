from functools import reduce
from numba import cuda
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


cudareduced_gcd = cuda.reduce(math.gcd)


# @cuda.jit
# def find_gcds(ar, gcds):
#     pos = cuda.grid(1)
#     if pos < gcds.shape[0]:
#         gcds[pos] = cuda.reduce(math.gcd, ar[pos])
def find_gcds(ar):
    gcds = np.empty(ar.shape[0], dtype=int)
    for i in range(ar.shape[0]):
        gcds[i] = reduce(math.gcd, ar[i])
    return gcds


@cuda.jit
def simplify(ar, gcds):
    row, col = cuda.grid(2)
    if row < ar.shape[0] and col < ar.shape[1]:
        d = gcds[row]
        if d != 1 and d != 0:
            ar[row, col] = ar[row, col] // d


def solv(input_arr):
    # pre_basis_main = create_pre_basis(input_arr[0])
    pre_basis_main = cuda.device_array(((len(input_arr[0])-1), len(input_arr[0])), dtype=int)

    threadsperblock = 256
    blockspergrid = math.ceil((len(input_arr[0])-1)/threadsperblock)

    create_pre_basis[blockspergrid, threadsperblock](np.array(input_arr[0], dtype=int), pre_basis_main)
    for Li in range(1, len(input_arr)):
        subresult_global_mem = cuda.device_array(pre_basis_main.shape[0], dtype=int)
        subinput1 = np.array(input_arr[Li], dtype=int)
        threadsperblock = 256
        blockspergrid = math.ceil(pre_basis_main.shape[0]/threadsperblock)
        cuda_substitute[blockspergrid, threadsperblock](subinput1, pre_basis_main, subresult_global_mem)
        #debug_subresult_global_mem = subresult_global_mem.copy_to_host()
        # Y = subresult_global_mem.copy_to_host().tolist()

        #pre_basis_Y = create_pre_basis(Y)
        pre_basis_Y = cuda.device_array((subresult_global_mem.shape[0]-1, subresult_global_mem.shape[0]), dtype=int)
        threadsperblock = 256
        blockspergrid = math.ceil((subresult_global_mem.shape[0] - 1) / threadsperblock)
        create_pre_basis[blockspergrid, threadsperblock](subresult_global_mem, pre_basis_Y)

        #debug_pre_basis_Y = pre_basis_Y.copy_to_host()
        #mult_pre_basis_Y = np.array(pre_basis_Y, dtype=int)
        mult_result_global_mem = cuda.device_array((pre_basis_Y.shape[0], pre_basis_main.shape[1]), dtype=int)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(pre_basis_Y.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(pre_basis_main.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        multiply_pre_basis[blockspergrid, threadsperblock](pre_basis_Y, pre_basis_main, mult_result_global_mem)

        pre_basis_main = mult_result_global_mem

        #gcds_global_mem = cuda.device_array(pre_basis_main.shape[0], dtype=int)

        # threadsperblock = 256
        # blockspergrid = math.ceil(pre_basis_main.shape[0]/threadsperblock)
        # find_gcds[blockspergrid, threadsperblock](pre_basis_main, gcds_global_mem)
        gcds = find_gcds(pre_basis_main.copy_to_host())
        gcds_global_mem = cuda.to_device(gcds)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(pre_basis_main.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(pre_basis_main.shape[1] /threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        simplify[blockspergrid, threadsperblock](pre_basis_main, gcds_global_mem)

        #pre_basis_main = simplify(pre_basis_main)
        #a = pre_basis_main.copy_to_host()
    return pre_basis_main.copy_to_host()


def control_main(input_name):
    return solv(read_file(input_name))

def timetest():
    solv(read_file(file))

if __name__ == "__main__":
    file = 'multiplied100x300.txt'
    print(solv(read_file(file)))
    import timeit
    print(timeit.timeit("timetest()", setup="from __main__ import timetest", number=100))
