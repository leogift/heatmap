import torch
import math

def meshgrid2d(B, X, Y):
    # returns a meshgrid sized B x X x Y
    grid_x = torch.linspace(0.0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, X, 1])
    grid_x = grid_x.repeat(B, 1, Y)

    grid_y = torch.linspace(0.0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, 1, Y])
    grid_y = grid_y.repeat(B, X, 1)

    return grid_x, grid_y

def gaussian_radius(feature_size, min_overlap=0.985, stride=32):
    height, width = feature_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3) * stride / 32 + 0.5

def special_multiples(input_num, base_num):
    multiples = math.ceil(input_num / base_num)
    return int(multiples * base_num)
