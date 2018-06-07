import os
import timeit
import solver
import csv


input_directory = './Input1200/'


def get_input_size(input_name):
    input_arr = solver.read_file(input_directory + input_name)
    return [len(input_arr), len(input_arr[0])]


for filename in os.listdir(input_directory):
    main_output_file = open('./Output/_Python_CUDA_31beta.csv', 'a', newline='')
    solver.control_main(input_directory + filename)
    avg_time = timeit.timeit("solver.control_main(input_directory + filename)", globals=globals(), number=10)/10.0
    input_size = get_input_size(filename)
    writer = csv.writer(main_output_file)
    writer.writerow([str(input_size[0]), str(input_size[1]), str(avg_time)])
    print(input_size[0], input_size[1])
    main_output_file.close()
    # with open("./Output/" + filename[:-4] + '(out).txt', 'w') as output_file:
    #     for vector in result:
    #         output_file.write(str(vector)+'\n')
