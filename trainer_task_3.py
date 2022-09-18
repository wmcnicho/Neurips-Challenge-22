import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import pandas as pd
import torch
import os
import argparse
import heapq
# from utils import open_json, dump_json, compute_auc, compute_accuracy
import time

from dataset_task_3 import create_dataset
# from model_task_3 import PermutedDKT
DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_information = []

# def pivot_df(df, values):
#     """
#     Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
#     unobserved.
#     """    
#     data = df.pivot_table(index='QuestionId', columns='UserId', values=values, sort=False)
#     # data = df.pivot_table(index='QuestionId', columns='UserId', values=values, fill_value=0, sort=False)

#     print("Number of questions: ", data.shape[0])
#     print("Number of students: ", data.shape[1])
#     # data.replace(np.nan,0)
#     if values == 'ConstructId':
#         data.fillna(0, inplace=True)
#         data = data.astype(int)
#         if DEBUG:
#             # Check how many students have answered questions.
#             stutdent_count_data = data.astype(bool).sum(axis=0)
#             p = plt.plot(stutdent_count_data.index, stutdent_count_data.values)
#             plt.show()
#     elif values == "IsCorrect":
#         data.replace(to_replace=0.0, value=-1.0, inplace=True)
#         data.fillna(0, inplace=True)
#         data = data.astype(int)

#     if DEBUG:
#         print(values, "Table:")
#         print(data)

#     return data

# def CreateDataset()
if __name__ == "__main__":

    seedNum = 221
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

    data_path = os.path.normpath('data_task_3/Task_3_dataset/checkins_lessons_checkouts_training.csv')
    create_dataset(data_path)
    # data_df = pd.read_csv(data_path)
    # # data_df = data_df.astype('category')
    # checkin_df = data_df[data_df['Type'] == 'Checkin'] # Only consider CheckIn.
    
    # construct_data = pivot_df(checkin_df, 'ConstructId') # feature
    # answer_data = pivot_df(checkin_df, 'IsCorrect') # label

    # train_data_path = os.path.normpath('data_task_3/train_task_3.csv')        
    # valid_data_path = os.path.normpath('data_task_3/valid_task_3.csv')
    # valid_df = pd.read_csv(valid_data_path)


    # question_meta = open_json('data_task_4/question_metadata_task_3_4.json')
    # train_data_path = os.path.normpath('data_task_4/train_task_4.csv')        
    # valid_data_path = os.path.normpath('data_task_4/valid_task_4.csv')
    # valid_df = pd.read_csv(valid_data_path)
    # valid_data = pivot_df(valid_df, 'AnswerValue')#n_student, 948:    1 to 4 and -1
    # valid_binary_data = pivot_df(valid_df, 'IsCorrect')  # n_student, 948: 1 to 0 and -1
    # train_df = pd.read_csv(train_data_path)
    # # n_student, 948:    1 to 4 and -1
    # train_data = pivot_df(train_df, 'AnswerValue')
    # # n_student, 948: 1 to 0 and -1
    # train_binary_data = pivot_df(train_df, 'IsCorrect')
    # train_dataset = FFDataset(train_data, train_binary_data,  question_meta)
    # valid_dataset = FFDataset(valid_data, valid_binary_data,  question_meta)
    # num_workers = 3
    # collate_fn =   ff_collate()
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, collate_fn=collate_fn, batch_size=16, num_workers=num_workers, shuffle=True, drop_last=True)
    # model = FFModel(hidden_dim=params.hidden_dim, dim=params.question_dim, concat_dim=params.concat_dim, concat_hidden_dim=params.concat_hidden_dim,dropout=params.dropout).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-8)
    # start_time = time.time()
    # for epoch in range(500):
    #     train_model()
    # end_time = time.time()
    # print("Time Elapsed: {} hours".format((end_time-start_time)/3600.))
