import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
import time

#import utils.plotting
#plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt



class BaseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(F.relu(x))
        return x


class BaseConv2d_factor(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BaseConv2d_factor, self).__init__()

        self.conv_1_kx1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride, (padding, 0))
        self.conv_2_1xk = nn.Conv2d(out_channels, out_channels, (1, kernel_size), stride, (0, padding))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_1_kx1(x)
        x = F.relu(x)
        x = self.conv_2_1xk(x)
        x = self.bn(F.relu(x))

        return x


class Incept_Module_A(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Incept_Module_A, self).__init__()

        self.conv_a_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b_3x3 = BaseConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c_5x5 = BaseConv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.bn = nn.BatchNorm2d(3*out_channels)

    def forward(self, x):
        x = torch.cat([self.conv_a_1x1(x), self.conv_b_3x3(x), self.conv_c_5x5(x)], dim=1)
        x = self.bn(F.relu(x))
        return x


class Incept_Module_B_factor(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Incept_Module_B_factor, self).__init__()

        self.conv_a1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_a2_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_a3_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_b1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b2_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_c1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv_d1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_d2_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d3_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_d4_3x3 = BaseConv2d_factor(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(4*out_channels)

    def forward(self, x):

        x_a = self.conv_a1_1x1(x)
        x_a = self.conv_a2_3x3(x_a)
        x_a = self.conv_a3_3x3(x_a)

        x_b = self.conv_b1_1x1(x)
        x_b = self.conv_b2_3x3(x_b)

        x_c = self.conv_c1_1x1(x)

        x_d = self.conv_d1_1x1(x)
        x_d = self.conv_d2_3x3(x_d)
        x_d = self.conv_d3_3x3(x_d)
        x_d = self.conv_d4_3x3(x_d)

        x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        x = F.relu(x)
        x = self.bn(x)

        return x


class Incept_Module_Bx(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Incept_Module_Bx, self).__init__()

        self.conv1_a = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_b = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_c = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = BaseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = BaseConv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(3*out_channels)

    def forward(self, x):
        x1 = self.conv1_a(x)
        x3 = self.conv3(self.conv1_b(x))
        x5 = self.conv5(self.conv1_c(x))
        x = torch.cat([x1, x3, x5], dim=1)
        x = F.relu(x)
        x = self.bn(x)
        return x


class Incept_Module_B(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Incept_Module_B, self).__init__()

        self.conv_a1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_a2_3x3 = BaseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_a3_3x3 = BaseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_b1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b2_3x3 = BaseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_c1_1x1 = BaseConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(3*out_channels)

    def forward(self, x):

        x_a = self.conv_a1_1x1(x)
        x_a = self.conv_a2_3x3(x_a)
        x_a = self.conv_a3_3x3(x_a)

        x_b = self.conv_b1_1x1(x)
        x_b = self.conv_b2_3x3(x_b)

        x_c = self.conv_c1_1x1(x)

        x = torch.cat([x_a, x_b, x_c], dim=1)
        x = F.relu(x)
        x = self.bn(x)

        return x


class Incept_Net(nn.Module):
    window_dim=52

    def __init__(self):
        super(Incept_Net, self).__init__()

        self.module_A = Incept_Module_A(4, 64)
        self.module_B1 = Incept_Module_B_factor(3*64, 64)
        self.module_B2 = Incept_Module_B_factor(4*64, 128)
        self.module_B3 = Incept_Module_B_factor(4*128, 128)

        self.mp1 = nn.MaxPool2d(4, stride=4, padding=0, dilation=1)
        self.mp2 = nn.MaxPool2d(3, stride=2, padding=0, dilation=1)
        self.mp3 = nn.MaxPool2d(6, stride=1, padding=0, dilation=1)

        self.bn1 = nn.BatchNorm2d(4*64)
        self.bn2 = nn.BatchNorm2d(4*128)
        self.bn3 = nn.BatchNorm2d(4*128)

        self.fc1 = nn.Conv2d(4*128, 512, kernel_size=1)
        self.fc2 = nn.Conv2d(512, 2, kernel_size=1)

        self.drop1 = nn.Dropout2d(p=0.5)

    def forward(self, x):

        do_print = False

        x = self.module_A(x)
        print_size(x, do_print)

        x = self.module_B1(x)
        x = self.bn1(F.relu(self.mp1(x)))
        print_size(x, do_print)

        x = self.module_B2(x)
        x = self.bn2(F.relu(self.mp2(x)))
        print_size(x, do_print)

        x = self.module_B3(x)
        x = self.bn3(F.relu(self.mp3(x)))
        print_size(x, do_print)

        x = self.drop1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        print_size(x, do_print)

        x = x.view(-1, 2)

        return x


def print_size(x, do_print):
    if do_print:
        print(x.size())



class Net(nn.Module):
    window_dim=52

    def __init__(self):
        super(Net, self).__init__()
        out_0 = 4
        out_1 = 64
        out_2 = 64
        out_3 = 128
        out_4 = 128
        out_5 = 256
        out_6 = 256
        out_7 = 256
        out_8 = 1024
        out_9 = 1024
        out_10 = 2

        self.conv1 = nn.Conv2d(out_0, out_1, 3, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(out_1, out_2, 3, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(out_2, out_3, 3, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(out_3, out_4, 3, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(out_4, out_5, 3, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(out_5, out_6, 3, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(out_6, out_7, 3, stride=1, padding=0, dilation=1)

        self.conv8 = nn.Conv2d(out_7, out_8, 1, stride=1, padding=0, dilation=1)
        self.conv9 = nn.Conv2d(out_8, out_9, 1, stride=1, padding=0, dilation=1)
        self.conv10 = nn.Conv2d(out_9, out_10, 1, stride=1, padding=0, dilation=1)

        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1)
        self.mp4 = nn.MaxPool2d(2, stride=2, padding=0, dilation=1)
        self.mp7 = nn.MaxPool2d(4, stride=1, padding=0, dilation=1)

        self.bn1 = nn.BatchNorm2d(out_1)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.bn4 = nn.BatchNorm2d(out_4)
        self.bn5 = nn.BatchNorm2d(out_5)
        self.bn6 = nn.BatchNorm2d(out_6)
        self.bn7 = nn.BatchNorm2d(out_7)

        self.drop1 = nn.Dropout2d(p=0.3)
        self.drop2 = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.mp2(F.relu(self.conv2(x))))

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(self.mp4(F.relu(self.conv4(x))))

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(self.mp7(F.relu(self.conv7(x))))

        x = self.drop1(F.relu(self.conv8(x)))
        x = self.drop2(F.relu(self.conv9(x)))
        x = self.conv10(x)
        x = x.view(-1, 2)

        return x


class Net2(nn.Module):
    window_dim=51

    def __init__(self):
        super(Net2, self).__init__()
        out_0 = 4
        out_1 = 64
        out_2 = 128
        out_3 = 256
        out_4 = 512
        out_5 = 512
        out_6 = 2

        self.conv1 = nn.Conv2d(out_0, out_1, 7, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(out_1, out_2, 5, stride=3, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(out_2, out_3, 5, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(out_3, out_4, 3, stride=1, padding=0, dilation=1)

        self.conv5 = nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1)

        self.bn1 = nn.BatchNorm2d(out_1)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.bn4 = nn.BatchNorm2d(out_4)

        self.drop1 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))

        x = self.drop1(F.relu(self.conv5(x)))
        x = self.conv6(x)
        x = x.view(-1, 2)

        return x


def init_weights_kaiming(m):
    if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
        torch.nn.init.kaiming_normal_(tensor=m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    return None

def init_weights_kaiming_high_std(m):
    if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
        torch.nn.init.kaiming_normal_(tensor=m.weight, a=100, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    return None

def init_weights_xavier(m):
    if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
        torch.nn.init.xavier_normal_(tensor=m.weight, gain=5/3)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    return None

def print_time(t):
    #torch.cuda.synchronize()
    #print(time.perf_counter() - t)
    #torch.cuda.synchronize()
    #return time.time()
    return time.perf_counter()



class BaseConv_separate(nn.Module):

    def __init__(self, c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, pool_kernel_size, pool_stride):
        super(BaseConv_separate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in*c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, groups=c_in, bias=True).apply(init_weights),
            nn.ReLU(),
            nn.Conv2d(c_in*c_out, c_out, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            nn.MaxPool2d(pool_kernel_size, pool_stride, padding=0, dilation=1)
        )

    def forward(self, x):
        return self.conv(x)




class BaseConv_short(nn.Module):

    def __init__(self, c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, pool_kernel_size,
                 pool_stride):
        super(BaseConv_short, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = conv_kernel_size
        self.num_filters = int(c_out ** 0.5)
        # self.num_filters = 10
        '''
        self.map = \
            torch.cat((
                torch.arange(start=0, end=c_in, dtype=torch.long).reshape((c_in, 1)).expand(c_in, c_out),
                torch.randint(high=self.num_filters, size=(c_in, c_out), dtype=torch.long)
            )).reshape(2, c_in, c_out).requires_grad_(False)
        '''
        self.map = torch.randint(high=self.num_filters, size=(c_in * c_out, 1), dtype=torch.long).requires_grad_(False)
        self.conv_short = nn.Conv2d(
            c_in, c_in * self.num_filters, conv_kernel_size, conv_stride, conv_padding, conv_dilation, groups=c_in,
            bias=False
        ).apply(init_weights)
        print(self.conv_short.weight.size())

        '''
        self.conv_short.weight = nn.Parameter(self.conv_short.weight.reshape(self.c_in, self.num_filters, conv_kernel_size, conv_kernel_size))
        self.conv_short.weight = nn.Parameter(self.conv_short.weight[self.map[0], self.map[1], :, :])
        self.conv_short.weight = nn.Parameter(self.conv_short.weight.reshape(-1, 1, conv_kernel_size, conv_kernel_size))
        '''

        # self.conv_short.weight = nn.Parameter(self.conv_short.weight[self.map.reshape(-1, 1), 0, :, :])
        self.conv_short.weight = nn.Parameter(self.conv_short.weight[self.map, 0, :, :])

        # self.map = torch.randint(high=self.num_filters, size=(c_in, c_out), dtype=torch.uint8).requires_grad_(False)
        self.batch_max = nn.Sequential(
            # nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            nn.MaxPool2d(pool_kernel_size, pool_stride, padding=0, dilation=1)
        )

    def forward(self, x):
        # self.conv_short.weight = nn.Parameter(self.conv_short.weight.reshape(self.c_in, self.num_filters, self.kernel_size, self.kernel_size))
        # self.conv_short.weight = self.conv_short.weight[self.map[0], self.map[1], :, :]
        # print(self.conv_short.weight.size())

        # print('map')
        # print(self.conv_short.weight.reshape(self.c_in, self.c_out, 1, 3, 3).size())
        # print(self.conv_short.weight.reshape(self.c_in, self.c_out, 1, 3, 3)[self.map[0, 1, 2], self.map[1, 1, :][self.map[1, 1, :] == 3], 0, 1, 2])
        # print('\n\n')

        t = time.perf_counter()
        y = self.conv_short(x)
        t = print_time(t)
        # print(y.size())
        y = y.reshape(y.size()[0], self.c_in, self.c_out, y.size()[2], y.size()[3])
        t = print_time(t)
        y = torch.sum(y, dim=1)
        t = print_time(t)
        # y = y.reshape(y.size()[0], self.c_in, self.num_filters, y.size()[2], y.size()[3])

        # y = y[:, self.map[0], self.map[1], :, :]
        # y = y.sum(dim=1)

        y = self.batch_max(y)
        t = print_time(t)

        # print(y.size())

        ''' 
        for i in range(self.c_in):
            if i == 0:
                #z = y[:, self.map[0, i, :].long(), self.map[1, i, :].long(), :, :]
                z = y[:, i, self.map[i, :].long(), :, :]
                #print(z.size())
            else:
                #z += y[:, self.map[0, i, :].long(), self.map[1, i, :].long(), :, :]
                #z = z.add(y[:, i, self.map[i, :].long(), :, :])
                z.add_(y[:, i, self.map[i, :].long(), :, :])
                #print(z.size())

        z = self.batch_max(z)
        return z
        '''

        return y

class BaseConv(nn.Module):

    def __init__(self, c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, pool_kernel_size, pool_stride):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, bias=False),#.apply(init_weights_kaiming),
            nn.BatchNorm2d(c_out),
            nn.PReLU(),
            #nn.AvgPool2d(pool_kernel_size, pool_stride)
            nn.MaxPool2d(pool_kernel_size, pool_stride, padding=0, dilation=1)
        )

    def forward(self, x):
        return self.conv(x)


class BaseConv_Net3(nn.Module):

    def __init__(self, c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, pool_kernel_size, pool_stride):
        super(BaseConv_Net3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, bias=False),#.apply(init_weights_kaiming),
            nn.BatchNorm2d(c_out),
            nn.PReLU(),
            #nn.AvgPool2d(pool_kernel_size, pool_stride)
            nn.MaxPool2d(pool_kernel_size, pool_stride, padding=0, dilation=1)
        )

    def forward(self, x):
        return self.conv(x)


# Sigmoid(log(input)), with trainable Sigmoid parameters - shared parameters for all input frequencies
class Sigmoid_Log_Shared_Freq(nn.Module):

    def __init__(self):
        super(Sigmoid_Log_Shared_Freq, self).__init__()
        #self.k = nn.Parameter(torch.Tensor([1810344.6]))
        #self.a = nn.Parameter(torch.Tensor([0.893]))
        self.eps = 1e-20
        self.k = nn.Parameter(torch.Tensor([2.58e-06]))
        self.a = nn.Parameter(torch.Tensor([-0.893]))

    def forward(self, input):
        x = input + self.eps
        #x[x > 0] = 1 / (1 + 1 / torch.pow(self.k * (x[x > 0]), self.a))
        x = 1 / (1 + self.k * torch.pow(x, self.a))
        return x

    # lr_k=e10
    # lr_a=0.001


# Sigmoid(log(input)), with trainable Sigmoid parameters - separate parameters for each input frequency
class Sigmoid_Log_Separate_Freq(nn.Module):

    def __init__(self):
        super(Sigmoid_Log_Separate_Freq, self).__init__()
        #init_k = 2.58e-06
        #init_a = -0.893
        self.eps = 1e-25
        #self.k = nn.Parameter(torch.Tensor(4).fill_(init_k).reshape(1, 4, 1, 1))
        #self.a = nn.Parameter(torch.Tensor(4).fill_(init_a).reshape(1, 4, 1, 1))
        #self.k = nn.Parameter(torch.Tensor([1.2e-04, 7.35e-05, 3.5e-05, 2.28e-05]).reshape(1, 4, 1, 1))
        self.k = nn.Parameter(torch.Tensor([1.84e-04, 2.2e-04, 1.6e-04, 4.0e-05]).reshape(1, 4, 1, 1))
        self.a = nn.Parameter(torch.Tensor([-1.25, -0.62, -0.62, -0.66]).reshape(1, 4, 1, 1))

    def forward(self, input):
        x = input + self.eps
        #x[x > 0] = 1 / (1 + 1 / torch.pow(self.k*(x[x > 0]), self.a))
        #x = 1 / (1 + 1 / torch.pow(self.k*x, self.a))
        x = 1 / (1 + self.k * torch.pow(x, self.a))
        return x


class Net3_new(nn.Module):
    window_dim=54

    def __init__(self):
        super(Net3_new, self).__init__()

        out_0 = 4
        out_1 = 128
        out_2 = 256
        out_3 = 1024
        out_4 = 2048
        out_5 = 2048
        out_6 = 2

        #self.preprocess = Sigmoid_Log_Shared_Freq()
        self.preprocess = Sigmoid_Log_Separate_Freq()

        self.main = nn.Sequential(
            BaseConv_Net3(out_0, out_1, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_1, out_2, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_2, out_3, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_3, out_4, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=3, pool_stride=1),
            nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1, bias=True),#.apply(init_weights),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1, bias=True)#.apply(init_weights)
        )


    def forward(self, x):

        #input_size = x.size()
        #x = x.permute(1, 0, 2, 3).view(input_size[1], -1)
        x = self.preprocess(x)
        #x = x.view(input_size).permute(1, 0, 2, 3)
        x = self.main(x)
        #x = x.view(-1, 2)
        return x



class Net3(nn.Module):
    window_dim=54

    def __init__(self):
        super(Net3, self).__init__()

        pre_1 = 16
        pre_2 = 16

        out_0 = 4
        out_1 = 128
        out_2 = 256
        out_3 = 1024
        out_4 = 2048
        out_5 = 2048
        out_6 = 2





        self.main = nn.Sequential(
            BaseConv_Net3(out_0, out_1, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_1, out_2, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_2, out_3, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv_Net3(out_3, out_4, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=3, pool_stride=1),
            nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1, bias=True),#.apply(init_weights),
            nn.PReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1, bias=True)#.apply(init_weights)
        )


    def forward(self, x):

        '''
        input_size = x.size()
        x = x.view(-1, 1)
        #x = torch.log(1+x)
        x = self.preprocess1(x)
        x = self.preprocess2(x)
        x = self.preprocess3(x)
        x = x.view(input_size[0], input_size[1], -1)

        x = x.sub(x.min(dim=2, keepdim=True)[0])
        max = x.max(dim=2, keepdim=True)[0]
        max[max == 0] = 1
        x = x.div(max)

        #x = x.div(torch.max(x, dim=2, keepdim=True))
        #x = x.div(torch.std(x, dim=2, keepdim=True) + 1e-30)


        x = x.view(input_size)
        '''

        x = self.main(x)
        x = x.view(-1, 2)
        return x


class Preprocess(nn.Module):

    def __init__(self, c_in, c_out, init_weights):
        super(Preprocess, self).__init__()

        if init_weights == 'kaiming':
            _init_weights = init_weights_kaiming
        elif init_weights == 'xavier':
            _init_weights = init_weights_xavier
        else:
            _init_weights = id

        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out)#.apply(_init_weights)
            #nn.BatchNorm1d(c_out)
        )

    def forward(self, x):
        return self.fc(x)


class PrintLayer(nn.Module):
    def __init__(self, layer):
        super(PrintLayer, self).__init__()
        self.layer = layer

    def forward(self, x):
        y = x.detach()
        a = torch.sum(y == 0).item() / torch.numel(y)
        if a > 0.9:
            print(self.layer + ': ' + str(a))
        return x


class Net3_preprocess(nn.Module):
    window_dim = 54



    def __init__(self):
        super(Net3_preprocess, self).__init__()

        pre_0 = 1
        pre_1 = 2
        pre_2 = 1

        out_0 = 4
        out_1 = 128
        out_2 = 256
        out_3 = 1024
        out_4 = 2048
        out_5 = 2048
        out_6 = 2

        self.preprocess = nn.Sequential(
            Preprocess(pre_0, pre_1, init_weights='kaiming'),
            nn.PReLU(),
            PrintLayer('layer_pre_1'),
            Preprocess(pre_1, pre_2, init_weights='kaiming')
        )

        self.main = nn.Sequential(
            BaseConv(out_0, out_1, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            PrintLayer('layer_conv_1'),
            BaseConv(out_1, out_2, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            PrintLayer('layer_conv_2'),
            BaseConv(out_2, out_3, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            PrintLayer('layer_conv_3'),
            BaseConv(out_3, out_4, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=3, pool_stride=1),
            PrintLayer('layer_conv_4'),
            nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1, bias=True),#.apply(init_weights_kaiming),
            nn.PReLU(),
            PrintLayer('layer_fc_1'),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1, bias=True),#.apply(init_weights_kaiming),
            PrintLayer('layer_fc_2')
        )

    def forward(self, x):

        print(self.preprocess[0].fc[0].weight)
        input_size = x.size()
        x = x.view(-1, 1) # Reshape for nn.Linear
        x = self.preprocess(x)
        #x = torch.log(x + 1e-10)

        #y = torch.tanh(x)
        y = x.view(input_size[0], input_size[1], -1)
        #print(y.size())
        y = y.sub(y.mean(dim=2, keepdim=True))
        y_std = y.std(dim=2, keepdim=True)
        #print(y_std)
        y_std_zero_to_one = 1.0 * (y_std == 0).type_as(y_std).to(torch.device('cuda:3'))
        #print(y_std.size())
        #print(y_std_zero_to_one.size())
        y_std = torch.max(input=y_std, other=y_std_zero_to_one)

        print(torch.min(y_std.cpu()), torch.max(y_std.cpu()))

        if torch.isnan(y.cpu()).any().item() == 2:
            with torch.no_grad():
                u = torch.linspace(0, 1000, 1000) + torch.ones(1000).uniform_(0.0, 1.0)
                u = u * 1 / 50
                u = torch.pow(u, 3)
                u = u.view(-1, 1).to(torch.device('cuda:3'))

                plt.gcf()
                plt.plot(u.detach().cpu().numpy(), self.preprocess(u).detach().cpu().numpy())
                plt.show()
                plt.clf()

        y = y * 1/(y_std + 1e-10)

        print(torch.min(y.cpu()), torch.max(y.cpu()))
        print('\n')

        #print(y_std)
        y = y.view(input_size)
        y = self.main(y)
        y = y.view(-1, 2)

        '''
        u = torch.linspace(0, 20, 1000) # Replace linspace with random values
        u = torch.pow(u, 3)
        u = u.view(-1, 1).to(torch.device('cuda:3'))
        z = self.preprocess(u)
        # print(y.cpu())

        z = z[1:] - z[:-1]
        z = torch.relu(-z)

        # print(y.cpu())
        #target = torch.zeros(y.size(), requires_grad=False).to(torch.device('cuda:3'))
        #loss_preprocess = F.l1_loss(y, target, reduction='sum')
        #return y, z
        '''

        return y

    #'''
    def preprocess_loss(self):

        x = torch.linspace(0, 1000, 1000) + torch.ones(1000).uniform_(0.0, 1.0)
        x = x * 1/50
        x = torch.pow(x, 3)
        x = x.view(-1, 1).to(torch.device('cuda:3'))
        y = self.preprocess(x)
        y = y[1:] - y[:-1]
        y = torch.relu(-y)

        z = torch.dot((x[1:] - x[:-1]).reshape(-1), y.reshape(-1))
        z = z * 1/torch.numel(x)
        #print(z.cpu())

        #target = torch.zeros(y.size(), requires_grad=False).to(torch.device('cuda:3'))
        #return F.l1_loss(y, target)
        return z
    #'''


if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Net3_new()
    model.to(device)

    '''
    x = np.random.rand(10, 4, 54, 54)
    x[0,0,0,0] = 0.0
    x = torch.Tensor(x).float()
    x = x.to(device)
    model = Net3_new()
    model.to(device)
    out = model(x)
    '''

    #'''
    y = np.random.rand(2, 4, 2, 2)
    #y[0, 0, 0, 0] = 0.0
    y = torch.Tensor(y).float()
    y = y.to(device)
    print(y.size())
    #print(y)
    pre = model.preprocess(y)
    #print(pre)
    #model.preprocess_loss()
    #'''

    #print(model.preprocess.weight.data.cpu().numpy())
    #print(model.preprocess.parameters())
    print(model.preprocess.k.data)
    print(model.preprocess.k.data[0,0,0,0].item())



    #print(model.main[0].conv[0].weight.cpu().size())

    #print(model.preprocess[0].fc[0].weight.detach().cpu().numpy())
    #print(model.preprocess[2].fc[0].weight.detach().cpu().numpy())
    #print(model.preprocess[4].fc[0].weight.detach().cpu().numpy())