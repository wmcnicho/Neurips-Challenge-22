import math
from symbol import xor_expr
import torch
import torch.nn as nn
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
torch.manual_seed(20)
import numpy as np

def generate_labels(features: torch.tensor):
    size = params.num_constructs

    W_ir = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_ir, a=math.sqrt(size), mode='fan_out')
    b_ir = torch.randn(size, dtype = torch.double)
    W_hr = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_hr, a=math.sqrt(size), mode='fan_out')
    b_hr = torch.randn(size, dtype = torch.double)
    W_iz = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_iz, a=math.sqrt(size), mode='fan_out')
    b_iz = torch.randn(size, dtype = torch.double)
    W_hz = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_hz, a=math.sqrt(size), mode='fan_out')
    b_hz = torch.randn(size, dtype = torch.double)
    W_in = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_in, a=math.sqrt(size), mode='fan_out')
    b_in = torch.randn(size, dtype = torch.double)
    W_hn = torch.empty(size, size, dtype = torch.double)
    nn.init.kaiming_normal_(W_hn, a=math.sqrt(size), mode='fan_out')
    b_hn = torch.randn(size, dtype = torch.double)

    p_matrix = torch.eye(size, dtype = torch.double)
    lower = torch.tril(torch.ones(size, size))
    lower = lower.to(torch.double)

    output_lower = torch.matmul(torch.matmul(p_matrix, lower), p_matrix.t()).t()

    W_ir = W_ir * lower
    W_hr = W_hr * lower
    W_iz = W_iz * lower
    W_hz = W_hz * lower
    W_in = W_in * lower
    W_hn = W_hn * lower

    w = 3 * torch.randn(size)
    b = 0.5 * torch.randn(size)

    def calulate_hidden(x, hidden, construct_id):
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        mask = torch.zeros(size, dtype = torch.double)
        mask[construct_id] = 1.0
        r_t = sigmoid(torch.matmul(x, W_ir) + b_ir * mask + torch.matmul(hidden, W_hr) + b_hr * mask)
        z_t = sigmoid(torch.matmul(x, W_iz) + b_iz * mask + torch.matmul(hidden, W_hz) + b_hz * mask)
        n_t = tanh(torch.matmul(x, W_in) +  b_in * mask + r_t * (torch.matmul(hidden, W_hn) + b_hn * mask))
        hy =  (1.0 - z_t) * n_t + z_t * hidden
        return hy

    def pred(hidden, idx):
        y_hat = torch.sigmoid(hidden[idx] * w[idx] + b[idx])
        print("y_hat:", y_hat)
        return y_hat

    def generate_y_vals(x_vals):
        hidden = torch.randn(size, dtype = torch.double)
        y_vals = torch.tensor([], dtype = torch.double)
        y_0 = pred(hidden, x_vals[0])
        if y_0.item() >= 0.5:
            y_vals = torch.concat((y_vals, torch.tensor([1.0])))
        else:
            y_vals = torch.concat((y_vals, torch.tensor([-1.0])))
        
        for idx, val in enumerate(x_vals):
            x = torch.zeros(size, dtype = torch.double)
            x[val] = y_vals[idx]
            # print("x:", x)
            # x[val] = 1.0
            hidden = calulate_hidden(x, hidden, val)
            # print("hidden state:", hidden)
            if idx + 1 == len(x_vals): break
            y_hat = pred(hidden, x_vals[idx+1])
            if y_hat.item() >= 0.5:
                y_vals = torch.concat((y_vals, torch.tensor([1.0])))
            else:
                y_vals = torch.concat((y_vals, torch.tensor([-1.0])))
            # print(hidden)
        return y_vals    

    number_students = 20
    number_questions = 10
    x_batch = np.random.randint(0, 5, size = (number_students, number_questions))
    y_batch = torch.tensor([], dtype = torch.double)
    for idx, x_val in enumerate(x_batch):
        print("-----------", idx)
        # print("x_vals", x_val)
        y_batch = torch.concat((y_batch, generate_y_vals(x_val)))

    y_batch = torch.reshape(y_batch, (number_students, number_questions))

    for idx, val in enumerate(x_batch):
        print("x:", val)
        print("y:", y_batch[idx]) 

    return y_batch
        

# x = [0,1,2,3,4]
# print(generate_y_vals(x))
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ML')

    parser.add_argument('-B', '--batch_size', type=int ,default=1, help='batch size')
    parser.add_argument('-C', '--num_constructs', type=int, default=5, help='number of constructs')
    parser.add_argument('-Q', '--num_questions', type=int, default=10, help='number of questions')
    parser.add_argument('-S', '--num_students', type=int, default=20, help='number of students')
    parser.add_argument('-T', '--temperature', type=int ,default=100, help='temperature')
    parser.add_argument('-U', '--unroll', type=int, default=100, help='unroll')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('-E', '--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-P', '--permutation', action="store_true", help='permute construct order')

    params = parser.parse_args()

    file_name = [params.num_constructs, params.num_questions, params.num_students, params.temperature, params.unroll, params.learning_rate, params.num_epochs]
    file_name =  [str(d) for d in file_name]
    params.file_name = '_'.join(file_name)
    seed_num = 35
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    # random.seed(seed_num)

    print(f"# of constructs: {params.num_constructs}\n# of questions: {params.num_questions}\n# of students: {params.num_students}")
    features = torch.randint(0, params.num_constructs, (params.num_students, params.num_questions))
    # labels = generate_labels(features.t())
    labels = generate_labels(features)

    print(labels)

# x = [0,1,2,3,4]
# print(generate_y_vals(x))
    
    
    







