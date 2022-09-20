import numpy as np
import pandas as pd
import torch
from torch.utils import data
import time
import torch
import random
# from utils import open_json, dump_json

def pivot_df(df, values, DEBUG=False):
    """
    Convert dataframe of question and answer records to pivoted array, filling in missing columns  
    if some questions are unobserved.
    """    
    data = df.pivot_table(index='UserId', columns='QuestionId', values=values, sort=False)
    if values == 'ConstructId':
        data.fillna(0, inplace=True)
        data = data.astype(int)
        if DEBUG:
            # Check how many students have answered questions.
            stutdent_count_data = data.astype(bool).sum(axis=0)
            # p = plt.plot(stutdent_count_data.index, stutdent_count_data.values)
            # plt.show()
    elif values == "IsCorrect":
        data.replace(to_replace=0.0, value=-1.0, inplace=True)
        data.fillna(0, inplace=True)
        data = data.astype(int)

    if DEBUG:
        print(values, "Table:")
        print(data)

    return data

class TaskDataset(data.Dataset):
    def __init__(self, X_data, Y_data):
        self.x = X_data
        self.y = Y_data
        unique_const_set = set()
        for q_id in X_data:
            c_list = X_data[q_id].unique()
            unique_const_set.update(c_list)
        self.unique_const_list = list(unique_const_set)

    def __len__(self):
        'Denotes the total number of students'
        return len(self.x)

    def __getitem__(self, idx):
        'Generates one sample of data: Answer of a student for whole questions'
        # sample = {'construct': torch.tensor([self.x.iloc[idx]]), 
        #         'answer': torch.tensor([self.y.iloc[idx]])}
        sample = {'construct': torch.tensor(self.x.iloc[idx].values), 
                'answer': torch.tensor(self.y.iloc[idx].values)}
        return sample


def create_dataset(data_path: str, DEBUG=False):
    data_df = pd.read_csv(data_path)
    checkin_df = data_df[data_df['Type'] == 'Checkin'] # Only consider CheckIn.
    X_data = pivot_df(checkin_df, 'ConstructId', False) # feature
    Y_data = pivot_df(checkin_df, 'IsCorrect', False) # label

    return TaskDataset(X_data, Y_data)

    # if DEBUG:
    #     print("length of the dataset is:", len(task_dataset))
    # train_size = int(0.8 * len(task_dataset))
    # valid_size = len(task_dataset) - train_size

    # return data.random_split(task_dataset, [train_size, valid_size])
    
class f_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
        #          'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
        #          'input_subjects': input_subjects, 'output_subjects': output_subjects, 'input_ans': torch.FloatTensor(input_ans)}
        B = len(batch)

        # input_labels  = torch.zeros(B, 2941).long()
        # output_labels = torch.zeros(B, 2941).long()
        # input_ans     = torch.ones(B, 2941).long()
        # input_mask    = torch.zeros(B,2941).long()
        # output_mask   = torch.zeros(B, 2941).long()
        # for b_idx in range(B):
        #     input_labels[b_idx, batch[b_idx]['input_question'].long()] =  batch[b_idx]['input_label'].long()
        #     input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
        #     input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
        #     output_labels[b_idx, batch[b_idx]['output_question'].long()] =  batch[b_idx]['output_label'].long()
        #     output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1

        # output = {'input_labels':input_labels, 'input_ans':input_ans, 'input_mask':input_mask, 'output_labels':output_labels, 'output_mask':output_mask}
        input_const  = torch.zeros(B, 2941).long()
        input_ans  = torch.ones(B, 2941).long()
        input_mask = torch.zeros(B,2941).long()
        print("DEBUG")
        for b_idx in range(B):
            print("b_idx: ", b_idx)
            print("=====================================")
            input_const[b_idx] = batch[b_idx]['construct']
            input_mask[b_idx] = batch[b_idx]['construct'].bool().int()
            input_ans[b_idx] = batch[b_idx]['answer']
            # torch.set_printoptions(threshold=40_000)
            print("=====================================")
        output = {'input_const' : input_const, 'input_mask' : input_mask, 'input_ans' : input_ans}
        return output