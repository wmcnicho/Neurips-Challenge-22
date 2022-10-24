import numpy as np
import argparse
import json
import numpy as np
import torch
import argparse
import pandas as pd
from predict_graph import PermutedDKT, PermutationMatrix, PermutedGru


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
    options = parser.parse_args()
    
    if device == torch.device('cpu'):
        model_load = torch.load(f'saved_models/{options.file_name}.pt', map_location=torch.device('cpu'))
    else:
        model_load = torch.load(f'saved_models/{options.file_name}.pt')
    try:
        p_matrix = model_load.gru.permuted_matrix.matrix
    except:
        p_matrix = model_load.module.gru.permuted_matrix.matrix

    sinkhorn_output = get_sinkhorn_output(p_matrix, options.temperature, options.unroll)
    np_matrix = sinkhorn_output.cpu().detach().numpy()
    if options.debug:
        np.save(f'./submissions/byproducts/sinkhorn_matrix_{options.file_name}.npy', np_matrix)

    argmax_search = search_argmax(np_matrix)
    if options.verbose:
        argmax_list_row = np.argmax(np_matrix, axis=1)
        argmax_list_col = np.argmax(np_matrix, axis=0)
        print('P Matrix by row', len(set(argmax_list_row)))
        print('P Matrix by col', len(set(argmax_list_col)))

    # Change the p-matrix to be ideal (row-wise argmax until the index has not been encountered) 
    p_matrix = np.zeros(np_matrix.shape)
    for row, col in enumerate(argmax_search):
        p_matrix[row][col] = 1
    if options.debug:
        np.save(f'./submissions/byproducts/p_matrix_{options.file_name}.npy', p_matrix)
    
    
    # NOTE: assuming perfect p-matrix
    # Read construct list
    with open("./serialized_torch/student_data_construct_list.json", 'rb') as fp:
        tot_construct_list = json.load(fp)

    tot_construct_list.append(0) # 0 is the padding construct
    if options.verbose:
        print('Original construct list', tot_construct_list)

    construct_arr = np.array(tot_construct_list)

    # get construct ordering
    construct_order = np.dot(p_matrix, construct_arr)
    construct_order_lst = construct_order.tolist()
    if options.verbose:
        print(construct_order_lst)  

    # read test data
    test_constructs = pd.read_csv('./data/Task_3_dataset/constructs_input_test.csv')['ConstructId'].tolist()
    solution_adj_matrix = np.zeros(shape=(len(test_constructs), len(test_constructs)))

    for i, row_cons in enumerate(test_constructs):
        row_pos = construct_order_lst.index(row_cons)
        for j, col_cons in enumerate(test_constructs):
            col_pos = construct_order_lst.index(col_cons)
            if col_pos >= row_pos:
                solution_adj_matrix[i][j] = 1 # construct j depends on construct i

    solution_adj_matrix_arr = np.array(solution_adj_matrix).astype(int)
    np.save(f'./submissions/adj_matrix_{options.file_name}.npy', solution_adj_matrix_arr)

if __name__ == "__main__":
    main()