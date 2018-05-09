import os
import time
import datetime
import timeit
import solver
import csv


def get_input_size(input_name):
    input_arr = solver.read_file('./Input/' + input_name)
    return [len(input_arr), len(input_arr[0])]

#main_output_file = open('./Output/_Main_output.csv', 'a', newline='')
for filename in os.listdir('./Input/'):
    main_output_file = open('./Output/_CUDA_Main_output.csv', 'a', newline='')
    # start_time = time.time()
    # result = solver.control_main('./Input/' + filename)
    # end_time = time.time()
    avg_time = timeit.timeit("solver.control_main('./Input/' + filename)", globals=globals(), number=4)/4.0
    input_size = get_input_size(filename)
    # string_to_main = input_size + ' - ' + str(end_time-start_time)
    # print(string_to_main + ' ' + str(datetime.datetime.today()))
    # main_output_file.write(string_to_main + '\n')
    writer = csv.writer(main_output_file)
    # writer.writerow([str(input_size[0]), str(input_size[1]), str(end_time-start_time)])
    writer.writerow([str(input_size[0]), str(input_size[1]), str(avg_time)])
    print(input_size[0], input_size[1])
    main_output_file.close()
    # with open("./Output/" + filename[:-4] + '(out).txt', 'w') as output_file:
    #     for vector in result:
    #         output_file.write(str(vector)+'\n')

#main_output_file.close()