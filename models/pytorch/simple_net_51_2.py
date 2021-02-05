import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
import time


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




def init_weights(m):
    if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
        torch.nn.init.kaiming_normal_(tensor=m.weight, a=0.0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    return None


class Preprocess(nn.Module):

    def __init__(self, c_in, c_out, non_linear=True):
        super(Preprocess, self).__init__()
        if non_linear:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out).apply(init_weights),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out).apply(init_weights)
            )

    def forward(self, x):
        return self.fc(x)

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



def print_time(t):
    #torch.cuda.synchronize()
    #print(time.perf_counter() - t)
    #torch.cuda.synchronize()
    #return time.time()
    return time.perf_counter()


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
            nn.Conv2d(c_in, c_out, conv_kernel_size, conv_stride, conv_padding, conv_dilation, bias=False).apply(init_weights),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            #nn.AvgPool2d(pool_kernel_size, pool_stride)
            nn.MaxPool2d(pool_kernel_size, pool_stride, padding=0, dilation=1)
        )

    def forward(self, x):
        return self.conv(x)

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


        #self.preprocess1 = Preprocess(1, pre_1, non_linear=True)
        #self.preprocess2 = Preprocess(pre_1, pre_2, non_linear=True)
        #self.preprocess3 = Preprocess(pre_2, 1, non_linear=False)

        '''
        self.conv1 = BaseConv_short(out_0, out_1, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2)
        self.conv2 = BaseConv_short(out_1, out_2, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2)
        self.conv3 = BaseConv(out_2, out_3, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2)
        self.conv4 = BaseConv(out_3, out_4, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=3, pool_stride=1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1, bias=True).apply(init_weights),
            nn.ReLU(),
            #nn.Dropout2d(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1, bias=True).apply(init_weights)
        )
        '''

        self.main = nn.Sequential(
            BaseConv(out_0, out_1, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv(out_1, out_2, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv(out_2, out_3, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=2, pool_stride=2),
            BaseConv(out_3, out_4, conv_kernel_size=3, conv_stride=1, conv_padding=0, conv_dilation=1, pool_kernel_size=3, pool_stride=1),
            nn.Conv2d(out_4, out_5, 1, stride=1, padding=0, dilation=1, bias=True).apply(init_weights),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(out_5, out_6, 1, stride=1, padding=0, dilation=1, bias=True).apply(init_weights)
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
        '''
        #print('\nconv1')
        x = self.conv1(x)  # 54 -> 52 -> 26
        #print('\nconv2')
        x = self.conv2(x)  # 26 -> 24 -> 12
        #print('\nconv3')
        x = self.conv3(x)  # 12 -> 10 -> 5
        #print('\nconv4')
        x = self.conv4(x)  # 5 -> 3 -> 1
        #print('\nfc1')
        t = time.perf_counter()
        x = self.fc1(x)
        t = print_time(t)
        #print('\nfc2')
        x = self.fc2(x)
        t = print_time(t)
        x = x.view(-1, 2)
        #print('x')
        return x
        '''
        x = self.main(x)
        x = x.view(-1, 2)
        return x


if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    x = np.random.rand(10, 4, 54, 54)
    x = torch.Tensor(x).float()
    x = x.to(device)
    model = Net3()
    model.to(device)
    out = model(x)
    #print(out)

