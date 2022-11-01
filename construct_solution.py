import numpy as np
import argparse
import json
import numpy as np
import torch
import argparse
import pandas as pd
from predict_graph import PermutedDKT, PermutationMatrix, PermutedGru
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sinkhorn_output(matrix, temperature, unroll):
    matrix_shape = matrix.shape[0]

    max_row = torch.max(matrix, dim=1).values.reshape(matrix_shape, 1)
    ones = torch.ones(matrix_shape, device=device).reshape(1, matrix_shape)

    matrix = torch.exp(temperature * (matrix - torch.matmul(max_row, ones)))
    
    for _ in range(unroll):
        matrix = matrix / torch.sum(matrix, dim=1, keepdim=True)
        matrix = matrix / torch.sum(matrix, dim=0, keepdim=True)
    
    return matrix

def search_argmax(matrix):
    taken_indices = {}
    index_lst = []
    for row in matrix:
        val_index_pair = {k:i for i, k in enumerate(row)}
        val_index_sorted = sorted(val_index_pair.items(), reverse=True)
        for val, index in val_index_sorted:
            try:
                res = taken_indices[index]
            except KeyError:
                index_lst.append(index)
                taken_indices[index] = 1
                break
    return index_lst

def main():
    parser = argparse.ArgumentParser(description='UMass 2022 casual ordering submission script')
    parser.add_argument('-f', '--file_name', type=str, default='final_10_22_20_25_44_batch_3_epoch_5_embed_3', help='Model file from training without file extension')
    parser.add_argument('-V', '--verbose', action=argparse.BooleanOptionalAction, help='Controls amount of printing')
    parser.add_argument('-D', '--debug', action=argparse.BooleanOptionalAction, help='Controls whether intermediate files are created during submission construction')
    parser.add_argument('-T', '--temperature', type=int, default=2, help='temperature of learned model')
    parser.add_argument('-U', '--unroll', type=int, default=5, help='unroll length of learned model')
    parser.add_argument('-L', '--tau', type=float, default=0.5, help='threshold for L matrix')
    options = parser.parse_args()
    
    if device == torch.device('cpu'):
        model_load = torch.load(f'saved_models/{options.file_name}.pt', map_location=torch.device('cpu'))
    else:
        model_load = torch.load(f'saved_models/{options.file_name}.pt')
    try:
        p_matrix = model_load.gru.permuted_matrix.matrix
        l_matrix = model_load.gru.permuted_matrix.lower
    except:
        p_matrix = model_load.module.gru.permuted_matrix.matrix
        l_matrix = model_load.module.gru.permuted_matrix.lower

    sinkhorn_output = get_sinkhorn_output(p_matrix, options.temperature, options.unroll)
    l_mask = torch.tril(torch.ones(l_matrix.shape[0], l_matrix.shape[1]))
    sigmoid_output = torch.sigmoid(l_matrix) * l_mask

    np_p_matrix = sinkhorn_output.cpu().detach().numpy()
    np_l_matrix = sigmoid_output.cpu().detach().numpy()

    np.set_printoptions(threshold=np.inf)
    if options.verbose:
        print("Trained L bar matrix: ", l_matrix)
        print("========="*10)
        print("L matrix: ", np_l_matrix)

    if options.debug:
        np.save(f'./submissions/byproducts/sinkhorn_p_matrix_{options.file_name}.npy', np_p_matrix)
        np.save(f'./submissions/byproducts/l_matrix_{options.file_name}.npy', np_l_matrix)

    argmax_search = search_argmax(np_p_matrix)
    if options.verbose:
        argmax_list_row = np.argmax(np_p_matrix, axis=1)
        argmax_list_col = np.argmax(np_p_matrix, axis=0)
        print('P Matrix by row', len(set(argmax_list_row)))
        print('P Matrix by col', len(set(argmax_list_col)))

    p_matrix = np.zeros(np_p_matrix.shape)
    for row, col in enumerate(argmax_search):
        p_matrix[row][col] = 1
    l_matrix = np.zeros(np_l_matrix.shape)

    max_element = 0
    min_element = 1
    for i in range(np_l_matrix.shape[0]):
        for j in range(np_l_matrix.shape[1]):
            if i > j:
                if np_l_matrix[i][j] > max_element:
                    max_element = np_l_matrix[i][j]
                elif np_l_matrix[i][j] < min_element:
                    min_element = np_l_matrix[i][j]
            elif i == j:
                np_l_matrix[i][j] = 1
    if options.verbose:
        print("Max element of L bar matrix: ", max_element)
        print("Min element of L bar matrix: ", min_element)
    l_matrix = (np_l_matrix > float(options.tau)).astype(float)

    if options.verbose:
        print("========="*10)
        print("Thresholded L matrix: ", l_matrix)

    if options.debug:
        np.save(f'./submissions/byproducts/threshold_{options.tau}_l_matrix_{options.file_name}.npy', l_matrix) # NOTE: thresholded l-matrix
        np.save(f'./submissions/byproducts/p_matrix_{options.file_name}.npy', p_matrix)
        p_matrix = np.load(f'./submissions/byproducts/p_matrix_{options.file_name}.npy') # NOTE: assuming perfect p-matrix

    # Read construct list
    with open("./serialized_torch/student_data_construct_list.json", 'rb') as fp:
        tot_construct_list = json.load(fp)

    tot_construct_list.append(0) # 0 is the padding construct
    if options.verbose:
        print('Original construct list', tot_construct_list)

    construct_arr = np.array(tot_construct_list)

    # Get construct ordering
    construct_order = np.dot(p_matrix, construct_arr)
    construct_order_lst = construct_order.tolist()
    if options.verbose:
        print(construct_order_lst)  

    # Read test data
    test_constructs = pd.read_csv('./data/Task_3_dataset/constructs_input_test.csv')['ConstructId'].tolist()
    solution_adj_matrix = np.zeros(shape=(len(test_constructs), len(test_constructs)))

    for i, row_cons in enumerate(test_constructs):
        row_pos = construct_order_lst.index(row_cons)
        for j, col_cons in enumerate(test_constructs):
            col_pos = construct_order_lst.index(col_cons)
            solution_adj_matrix[i][j] = l_matrix[row_pos][col_pos]

    solution_adj_matrix_arr = np.array(solution_adj_matrix).astype(int)

    directory = f"tau_{options.tau}"
    parent_dir = "./submissions/"
    path = os.path.join(parent_dir, directory)

    os.mkdir(path, 0o755)
    np.save(f'{path}/adj_matrix.npy', solution_adj_matrix_arr)
    shutil.make_archive(f'tau_{options.tau}', format='zip', root_dir=path)

if __name__ == "__main__":
    main()