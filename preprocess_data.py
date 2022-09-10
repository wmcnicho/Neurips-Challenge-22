from unittest import result
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F

def createDataset(filename: str):
    lessons_df = pd.read_csv(filename)
    # This is pretty aggresive cleaning, maybe others Types are useful
    checkin_df = lessons_df[lessons_df['Type'] == 'Checkin']
    # Clean out all the columns we don't need
    simple_df = checkin_df.iloc[:, [2, 5, 9]] # [3, ]
    simple_df.loc[simple_df["IsCorrect"] == 0, "IsCorrect"] = -1
    
    # A little dirty importing a module just for mode
    num_max_questions = stats.mode(simple_df["UserId"]).count[0]

    #C=num_constructs S=num_students
    result = torch.tensor([]) # [1] -> [S, C+1]
    for name, group in simple_df.groupby('UserId'):
        print(group)
        constructs = torch.tensor(group["ConstructId"].values.tolist()) # [C]
        correct = torch.tensor(group["IsCorrect"].values.tolist()) # [1]
        student = torch.tensor([constructs], [correct]) # [C +1]
        
        pad_needed = num_max_questions - student.shape[-1] # [p]
        # This pads the bottom of the matrix with pad_needed rows
        # Are we sure we need this?
        #padded = F.pad(input=student, pad=(0, 0, 0, pad_needed)) # [C+p, 1+p]
        #result_features = torch.stack(result_features, padded)
    return result
    
def main():
    tensor = createDataset('data/sample_data_lessons_small.csv')

if __name__ == "__main__":
    main()