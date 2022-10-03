import numpy as np
import pandas as pd
import json 

p_matrix = np.load('p_matrix_adaptive_small.npy') # NOTE: assuming perfect p-matrix

# TODO: change the p-matrix to be ideal (row-wise argmax until the index has not been encountered)

# Read construct list
with open("../serialized_torch/student_data_construct_list.json", 'rb') as fp:
    tot_construct_list = json.load(fp)

tot_construct_list.append(0) # 0 is the padding construct
print('Original construct list', tot_construct_list)

construct_arr = np.array(tot_construct_list)

# get construct ordering
construct_order = np.dot(p_matrix, construct_arr)
print(construct_order)
construct_order_lst = construct_order.tolist()

# #sanity check
# # identity_p_matrix = np.identity(len(tot_construct_list))
# sample_construct_order = np.dot(p_matrix, construct_arr)
# print(sample_construct_order)
# sample_construct_order_lst = sample_construct_order.tolist()

# read test data
test_constructs = pd.read_csv('../data/Task_3_dataset/constructs_input_test.csv')['ConstructId'].tolist()
solution_adj_matrix = np.zeros(shape=(len(test_constructs), len(test_constructs)))

for i, row_cons in enumerate(test_constructs):
    row_pos = construct_order_lst.index(row_cons)
    for j, col_cons in enumerate(test_constructs):
        col_pos = construct_order_lst.index(col_cons)
        if col_pos >= row_pos:
            solution_adj_matrix[i][j] = 1 # construct j depends on construct i

solution_adj_matrix_arr = np.array(solution_adj_matrix).astype(int)
np.save('adj_matrix_adaptive_small.npy', solution_adj_matrix_arr)
