import os
import pandas as pd
import torch
import json
from scipy import stats


def createDataset(filename: str):
    lessons_df = pd.read_csv(filename)
    # 1) Consider only type "Checkin"
    #   TODO: This is pretty aggresive cleaning, maybe others Types are useful
    checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']

    # 2) Clean out all the columns we don't need
    #   2: UserId, 5: IsCorrect, 9: ConstructId
    simple_df = checkin_df.iloc[:, [2, 5, 9]] # [S, 3]
    #   Substitue IsCorrect value of 0 to -1
    simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1 # [S, 3]
    
    # 3) Set number of questions (Q) as the most questions answered by a single student
    #   A little dirty importing a module just for mode
    num_of_questions = stats.mode(simple_df["UserId"]).count[0] # Q

    # 4) Create dataset in form of [S, 2, Q]
    #   S: number of unique students
    #   2: features and label
    #   Q: most questions answered by a single student
    
    tot_construct_set = set()
    result = torch.tensor([]) # [1] -> [S, 2, Q]
    for ctr, (user, user_info) in enumerate(simple_df.groupby('UserId')):
        if ctr % 100 == 0:
            print(ctr)
        # print("User: ", user, "\t", "User Info: ", user_info)
        
        # List of ConstructId and IsCorrect
        # C: number of constructs of each user
        constructs = user_info["ConstructId"].values.tolist() # [C]
        correct = user_info["IsCorrect"].values.tolist() # [C]

        tot_construct_set.update(constructs)

        # Pad for ConstructId that has not been dealt with a user
        num_of_constructs = len(constructs)
        pad_needed = num_of_questions - num_of_constructs # [P = Q - C]
        constructs += [0] * pad_needed # [Q]
        correct += [0] * pad_needed # [Q]

        # Build tensor
        student = torch.tensor([constructs, correct]) # [2, Q]
        student = torch.unsqueeze(student, 0) # [1, 2, Q]

        # Add to running result
        result = torch.cat([result, student]) # [i, 2, Q]

    # num_of_constructs = len(tot_construct_set)
    tot_construct_list = list(tot_construct_set)

    return result, tot_construct_list # [S, 2, Q]
    
def main():
    use_main = 1 # 1 for using main data, anything else for using sample data
    if use_main:
        data_path = 'data/Task_3_dataset/checkins_lessons_checkouts_training.csv'
        prefix_path = 'student_data'
    else:
        data_path = 'data/sample_data_lessons_small.csv'
        prefix_path = 'sample_student_data'
    student_data, tot_construct_list = createDataset(data_path) # [S, 2, Q] Faster and better for debugging
    print("num_of_constructs: ", len(tot_construct_list))

    transform_student_data = torch.transpose(student_data, 0, 2) # [S, 2, Q] --> [Q, 2, S]

    torch.save(transform_student_data, os.path.join('serialized_torch', prefix_path+'_tensor.pt'))
    with open(os.path.join('serialized_torch', prefix_path+'_construct_list.json'), "w") as fp:
        json.dump(tot_construct_list, fp)

    # student_data = createDataset('data/Task_3_dataset/checkins_lessons_checkouts_training.csv') # [S, 2, Q]
    features = transform_student_data[:, 0, :] # [Q, S]
    labels = transform_student_data[:, 1, :] # [Q, S]
    # print("labels: ", labels)
    # print("features: ", features)
    print('Features shape:', features.shape)
    print('Labels shape', labels.shape)


if __name__ == "__main__":
    # loaded_file = torch.load('serialized_torch/student_data_tensor.pt') # This is also an option
    main()