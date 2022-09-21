import math
import torch
import torch.nn as nn
# torch.manual_seed(20)
import numpy as np

size = 5

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

w = torch.randn(size) * 10
b = torch.randn(1)

def calulate_hidden(x, hidden, construct_id):
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    mask = torch.zeros(size, dtype = torch.double)
    mask[construct_id] = 1.0
    r_t = sigmoid(torch.matmul(x, W_ir) + b_ir * mask + torch.matmul(hidden, W_hr) + b_hr )
    z_t = sigmoid(torch.matmul(x, W_iz) + b_iz * mask + torch.matmul(hidden, W_hz) + b_hz )
    n_t = tanh(torch.matmul(x, W_in) +  b_in * mask + r_t * (torch.matmul(hidden, W_hn) + b_hn))
    hy =  (1.0 - z_t) * n_t + z_t * hidden
    return hy

def pred(hidden, idx):
    y_hat = torch.sigmoid(hidden[idx] * w[idx] + b)
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

# number_students = 100
# number_questions = 20
# x_batch = np.random.randint(0, 5, size = (number_students, number_questions))
# y_batch = torch.tensor([], dtype = torch.double)
# for idx, x_val in enumerate(x_batch):
#     print("-----------", idx)
#     # print("x_vals", x_val)
#     y_batch = torch.concat((y_batch, generate_y_vals(x_val)))

# y_batch = torch.reshape(y_batch, (number_students, number_questions))

# for idx, val in enumerate(x_batch):
#     print("x:", val)
#     print("y:", y_batch[idx]) 
    

x = [0,1,2,3,4]
print(generate_y_vals(x))
    
    
    







