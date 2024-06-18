import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

class square_approximation(nn.Module):
    def __init__(self, k_iter, input_size):
        super(square_approximation, self).__init__()
        self.input_size = input_size
        self.k_iter = k_iter
        self.g = []
        for _ in range(k_iter):
            hidden_1 = nn.Linear(input_size, 3 * input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.eye(input_size), torch.eye(input_size)), dim = 0))
            hidden_1.bias.data.copy_(torch.cat((torch.zeros(input_size), -torch.ones(input_size) / 2, -torch.ones(input_size)), dim = 0))

            hidden_2 = nn.Linear(3 * input_size, input_size)
            hidden_2.weight.data.copy_(torch.cat((2 * torch.eye(input_size), -4 * torch.eye(input_size), 2 * torch.eye(input_size)), dim = 1))
            hidden_2.bias.data.copy_(torch.zeros(input_size))
            self.g.append(nn.Sequential(hidden_1, nn.ReLU(), hidden_2))

        self.output = nn.Linear((k_iter + 1) * input_size, input_size)
        out_weights = torch.eye(input_size)
        for i in range(k_iter):
            out_weights = torch.cat(( out_weights, -torch.eye(input_size)/(2**(2 * i+2)) ), dim = 1)
        self.output.weight.data.copy_(out_weights)
        self.output.bias.data.copy_(torch.zeros(input_size))

    def forward(self, x):
        inputs = x
        temp = x
        for i in range(self.k_iter):
            temp = self.g[i](temp)
            inputs = torch.cat((inputs, temp), dim=0)
        output = self.output(inputs)
        return output


class cube_approximation(nn.Module):
    def __init__(self, k_iter, input_size):
        super(cube_approximation, self).__init__()
        self.input_size = input_size
        self.k_iter = k_iter

        #Building g^k(x); g(x)= I(x <=1/2) * 2x + I(x > 1/2) * (1 - 2x)
        self.g = []
        for _ in range(k_iter):
            hidden_1 = nn.Linear(input_size, 3 * input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.eye(input_size), torch.eye(input_size)), dim = 0))
            hidden_1.bias.data.copy_(torch.cat((torch.zeros(input_size), -torch.ones(input_size) / 2, -torch.ones(input_size)), dim = 0))

            hidden_2 = nn.Linear(3 * input_size, input_size)
            hidden_2.weight.data.copy_(torch.cat((2 * torch.eye(input_size), -4 * torch.eye(input_size), 2 * torch.eye(input_size)), dim = 1))
            hidden_2.bias.data.copy_(torch.zeros(input_size))
            self.g.append(nn.Sequential(hidden_1, nn.ReLU(), hidden_2))

        #Buiding s(x) = 1/2 + ReLU(x - 1/2)
        new_size = (k_iter + 1) * input_size
        hidden_1 = nn.Linear(new_size, new_size)
        hidden_1.weight.data.copy_(torch.eye(new_size))
        hidden_1.bias.data.copy_(-torch.ones(new_size)/2)

        hidden_2 = nn.Linear(new_size, new_size)
        hidden_2.weight.data.copy_(torch.eye(new_size))
        hidden_2.bias.data.copy_(torch.ones(new_size)/2)
        self.s = nn.Sequential(hidden_1, nn.ReLU(), hidden_2)

        #Building h_1(x)
        self.h = []
        hidden_1 = nn.Linear(new_size, 3 * new_size)
        hidden_1.weight.data.copy_(torch.cat((torch.eye(new_size), torch.eye(new_size), torch.eye(new_size)), dim = 0))
        hidden_1.bias.data.copy_(torch.cat((torch.zeros(new_size), -torch.ones(new_size) / 2, -torch.ones(new_size)), dim = 0))

        hidden_2 = nn.Linear(3 * new_size, new_size)
        hidden_2.weight.data.copy_(torch.cat((2 * torch.eye(new_size), -4 * torch.eye(new_size), 2 * torch.eye(new_size)), dim = 1))
        hidden_2.bias.data.copy_(torch.zeros(new_size))

        hidden_3 = nn.Linear(new_size, new_size - input_size)
        hidden_3.weight.data.copy_(torch.cat((torch.eye(new_size - input_size), torch.zeros((new_size - input_size, input_size))), dim = 1))
        hidden_3.bias.data.copy_(torch.zeros(new_size - input_size))
        self.h.append(nn.Sequential(hidden_1, nn.ReLU(), hidden_2, hidden_3))

        #Building h_i(x) = a_i(x) - (2^{i-1} + 1)b_i(x), i >=2
        #a_i(x) = h_{i-1}(g(x))
        #b_i = g^i(s(x))

        self.a = []
        temp_size = new_size - input_size
        while temp_size > input_size:
            hidden_1 = nn.Linear(temp_size, temp_size - input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.zeros((temp_size - input_size, input_size)), torch.eye(temp_size - input_size)), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(temp_size - input_size))
            self.a.append(hidden_1)
            temp_size -= input_size

        self.b = []
        #hidden_1 = nn.Linear(new_size, 3 * new_size)
       # hidden_1.weight.data.copy_(torch.cat((torch.eye(new_size), torch.eye(new_size), torch.eye(new_size)), dim = 0))
        #hidden_1.bias.data.copy_(torch.cat((torch.zeros(new_size), -torch.ones(new_size) / 2, -torch.ones(new_size)), dim = 0))

        #hidden_2 = nn.Linear(3 * new_size, new_size)
        #hidden_2.weight.data.copy_(torch.cat((2 * torch.eye(new_size), -4 * torch.eye(new_size), 2 * torch.eye(new_size)), dim = 1))
        #hidden_2.bias.data.copy_(torch.zeros(new_size))
        #self.g_b = (nn.Sequential(hidden_1, nn.ReLU(), hidden_2))

        temp_size = new_size
        while temp_size > input_size:
            hidden_1 = nn.Linear(temp_size, temp_size - input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(temp_size - input_size), torch.zeros((temp_size - input_size, input_size))), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(temp_size - input_size))

            temp_size -= input_size
            hidden_2 = nn.Linear(temp_size, 3 * temp_size)
            hidden_2.weight.data.copy_(torch.cat((torch.eye(temp_size), torch.eye(temp_size), torch.eye(temp_size)), dim = 0))
            hidden_2.bias.data.copy_(torch.cat((torch.zeros(temp_size), -torch.ones(temp_size) / 2, -torch.ones(temp_size)), dim = 0))

            hidden_3 = nn.Linear(3 * temp_size, temp_size)
            hidden_3.weight.data.copy_(torch.cat((2 * torch.eye(temp_size), -4 * torch.eye(temp_size), 2 * torch.eye(temp_size)), dim = 1))
            hidden_3.bias.data.copy_(torch.zeros(temp_size))

            self.b.append(nn.Sequential(hidden_1, hidden_2, nn.ReLU(), hidden_3))



        for k in range(k_iter - 1):
            hidden_1 = nn.Linear(2 * (k_iter - k - 1) * input_size, (k_iter - k - 1) * input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye((k_iter - k - 1) * input_size), -(2**(k + 1) + 1) * torch.eye((k_iter - k - 1) * input_size)), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros((k_iter - k - 1) * input_size))

            hidden_2 = nn.Linear((k_iter - k - 1) * input_size, 2 * (k_iter - k - 1) * input_size)
            hidden_2.weight.data.copy_(torch.cat((torch.eye((k_iter - k - 1) * input_size), -torch.eye((k_iter - k - 1) * input_size)), dim = 0))
            hidden_2.bias.data.copy_(torch.zeros(2 * (k_iter - k - 1) * input_size))

            hidden_3 = nn.Linear(2 * (k_iter - k - 1) * input_size, (k_iter - k - 1) * input_size)
            hidden_3.weight.data.copy_(torch.cat((torch.eye((k_iter - k - 1) * input_size), torch.eye((k_iter - k - 1) * input_size)), dim = 1))
            hidden_3.bias.data.copy_(torch.zeros((k_iter - k - 1) * input_size))

            self.h.append(nn.Sequential(hidden_1, hidden_2, nn.ReLU(), hidden_3))

        #h_ans is just first input_size elements of h_i
        self.h_ans = []
        temp_size = new_size - input_size
        while temp_size > 0:
            hidden_1 = nn.Linear(temp_size, input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.zeros((input_size, temp_size - input_size))), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            temp_size -=input_size
            self.h_ans.append(hidden_1)
        #output
        self.output = nn.Linear(2* new_size - input_size, input_size)

        output_weights = torch.eye(input_size)
        for k in range(k_iter):
            output_weights = torch.cat((output_weights, 3 * torch.eye(input_size)/ (2**(3 * k + 3))), dim = 1)

        for k in range(k_iter):
            output_weights = torch.cat((output_weights, -3 * torch.eye(input_size)/ (2**(3 * k + 2))), dim = 1)

        self.output.weight.data.copy_(output_weights)
        self.output.bias.data.copy_(torch.zeros(input_size))





    def forward(self, x):
        g_inputs = x
        temp = x
        for i in range(self.k_iter):
            temp = self.g[i](temp)
            g_inputs = torch.cat((g_inputs, temp), dim=0)

        s_inputs = self.s(g_inputs)

        #creating h_1
        h_inputs = self.h[0](g_inputs)
        h_basis = self.h_ans[0](h_inputs)
        #plt.plot(x, h_basis.detach().numpy())
        #plt.show()
        s_inputs = self.b[0](s_inputs)

        #creating h_i, i > 1
        for i in range(self.k_iter - 1):
            h_inputs = self.a[i](h_inputs)
            s_inputs = self.b[i + 1](s_inputs)
            h_inputs = self.h[i + 1](torch.cat((h_inputs, s_inputs), dim = 0))
            h_new = self.h_ans[i + 1](h_inputs)
            #plt.plot(x, h_new.detach().numpy())
            #plt.show()
            h_basis = torch.cat((h_basis, h_new), dim = 0)
        #print(g_inputs.shape, h_basis.shape)
        output = self.output(torch.cat((g_inputs, h_basis), dim = 0))
        return output


class exponent_approximation(nn.Module):
    def __init__(self, k_iter, input_size):
        super(exponent_approximation, self).__init__()
        self.input_size = input_size
        self.k_iter = k_iter

        #Building g^k(x); g(x)= I(x <=1/2) * 2x + I(x > 1/2) * (1 - 2x)
        self.g = []
        for _ in range(k_iter - 1):
            hidden_1 = nn.Linear(input_size, 3 * input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.eye(input_size), torch.eye(input_size)), dim = 0))
            hidden_1.bias.data.copy_(torch.cat((torch.zeros(input_size), -torch.ones(input_size) / 2, -torch.ones(input_size)), dim = 0))

            hidden_2 = nn.Linear(3 * input_size, input_size)
            hidden_2.weight.data.copy_(torch.cat((2 * torch.eye(input_size), -4 * torch.eye(input_size), 2 * torch.eye(input_size)), dim = 1))
            hidden_2.bias.data.copy_(torch.zeros(input_size))
            self.g.append(nn.Sequential(hidden_1, nn.ReLU(), hidden_2))

        #Building I_k(x)

        self.x_doubling = []
        for _ in range(k_iter - 2):
            hidden_1 = nn.Linear(input_size, input_size)
            hidden_1.weight.data.copy_(2 * torch.eye(input_size))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            self.x_doubling.append(hidden_1)


        self.I = []
        new_size = k_iter * input_size - input_size
        hidden_1 = nn.Linear(new_size, new_size)
        hidden_1.weight.data.copy_(torch.zeros(new_size, new_size))
        hidden_1.bias.data.copy_(torch.zeros(new_size))
        self.I.append(hidden_1)

        temp_size = new_size
        while temp_size > input_size:
            hidden_1 = nn.Linear(temp_size, temp_size - input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.zeros((temp_size - input_size, input_size)), torch.eye(temp_size - input_size) /2), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(temp_size - input_size))
            temp_size -= input_size
            self.I.append(hidden_1)

        self.I_functions = []

        for k in range(k_iter - 1):
            hidden_1 = nn.Linear(new_size - k * input_size, input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.zeros((input_size, new_size - (k + 1) * input_size))), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            self.I_functions.append(hidden_1)

        #Building Y_k(x)

        self.Y = []
        for k in range(k_iter - 1):
            hidden_1 = nn.Linear(2 * input_size, input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size) / 2**(k), 2 * torch.eye(input_size)), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            self.Y.append(hidden_1)

        #Building Psi_k(x)
        self.y_doubling = []
        for _ in range(k_iter - 1):
            hidden_1 = nn.Linear(input_size, input_size)
            hidden_1.weight.data.copy_(2 * torch.eye(input_size))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            self.y_doubling.append(hidden_1)

        new_size = k_iter * input_size
        self.Psi = []
        #Psi_1
        hidden_1 = nn.Linear(new_size, 3 * new_size)
        hidden_1.weight.data.copy_(torch.cat((torch.eye(new_size), torch.eye(new_size), torch.eye(new_size)), dim = 0))
        hidden_1.bias.data.copy_(torch.cat((torch.zeros(new_size), -torch.ones(new_size) / 2, -torch.ones(new_size)), dim = 0))

        hidden_2 = nn.Linear(3 * new_size, new_size)
        hidden_2.weight.data.copy_(torch.cat((2 * torch.exp(torch.tensor([1])) * torch.eye(new_size), -4 * torch.exp(torch.tensor([1])) * torch.eye(new_size), 2 * torch.exp(torch.tensor([1])) * torch.eye(new_size)), dim = 1))
        hidden_2.bias.data.copy_(torch.zeros(new_size))

        self.Psi.append(nn.Sequential(hidden_1, nn.ReLU(), hidden_2))

        temp_size = new_size
        while temp_size > input_size:
            hidden_1 = nn.Linear(temp_size, temp_size - input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.zeros((temp_size - input_size, input_size)) ,torch.eye(temp_size - input_size)), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(temp_size - input_size))
            self.Psi.append(hidden_1)
            temp_size -= input_size

        self.basis = []
        for k in range(k_iter):
            hidden_1 = nn.Linear(new_size - k * input_size, input_size)
            hidden_1.weight.data.copy_(torch.cat((torch.eye(input_size), torch.zeros((input_size, new_size - (k + 1) * input_size))), dim = 1))
            hidden_1.bias.data.copy_(torch.zeros(input_size))
            self.basis.append(hidden_1)

        #output
        self.output = nn.Linear(new_size + input_size, input_size)

        output_weights = (torch.exp(torch.tensor([1])) - 1) * torch.eye(input_size)
        for k in range(k_iter):
            output_weights = torch.cat((output_weights, ((torch.exp(torch.tensor([-1 / (2 ** ( k + 1))])) - (torch.exp((torch.tensor([-1 / (2 ** (k))])))/2 + 1/2)) * torch.eye(input_size))), dim = 1)
        self.output.weight.data.copy_(output_weights)
        self.output.bias.data.copy_(torch.ones(input_size))


    def calculate_Y(self, k, x):
        Y_basis = []
        g_temp = self.g[0](x)
        I_fake = x
        I_temp = x

        for i in range(self.k_iter - 2):
            I_temp = self.x_doubling[i](I_temp) - (I_temp >= 1/2).float()
            I_fake = torch.cat((I_fake, I_temp), dim = 0)

        indicators = (I_fake >= 1/2).float() / 4
        I_temp = self.I[0](I_fake)

        for i in range(1, k + 1):
            g_temp = self.g[i](g_temp)
            indicators = indicators[: -self.input_size]
            I_temp = self.I[i](I_temp) + indicators

        Y_temp = torch.cat((g_temp, self.I_functions[k](I_temp)), dim = 0)
        Y_ind = torch.cat((g_temp, 2 * self.I_functions[k](I_temp)), dim = 0)
        return self.Y[k](Y_temp), self.Y[k](Y_ind)

    def forward(self, x):
        #Building Psi_k(x)
        Y_inputs = x
        indicators = [torch.tensor([]) for _ in range(k_iter - 1)]
        for i in range(self.k_iter - 1):
            Y_new = x
            for j in range(i, -1, -1):
                Y_temp, Y_ind = self.calculate_Y(j, Y_new)
                indicators[j] = torch.cat((indicators[j], ((torch.exp(torch.tensor([1])) - 1) * (Y_ind == 2 * Y_new).float() + 1) ** (-1 / 2 ** (j + 1))), dim = 0)
                #Y_new = Y_temp * (Y_ind == 2 * Y_new).float() + (1 - Y_temp) * (Y_ind != 2 * Y_new).float()
                Y_new = Y_temp
            Y_inputs = torch.cat((Y_inputs, Y_new), dim = 0)

        Psi_basis = x
        Y_inputs = self.Psi[0](Y_inputs)
        new_basis_vector = self.basis[0](Y_inputs)
        #plt.plot(x, new_basis_vector.detach().numpy())
        #plt.show()

        Psi_basis = torch.cat((Psi_basis, new_basis_vector), dim = 0)
        for i in range(1, self.k_iter):
            Y_inputs = indicators[i - 1] * self.Psi[i](Y_inputs)
            new_basis_vector = self.basis[i](Y_inputs)
            #plt.plot(x, new_basis_vector.detach().numpy())
            #plt.show()
            Psi_basis = torch.cat((Psi_basis, new_basis_vector), dim = 0)

        #print(indicators[0])


        output = self.output(Psi_basis)


        return output