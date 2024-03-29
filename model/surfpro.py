# Parametric Networks for 3D Point Cloud Classification

import numpy as np
import torch
import torch.nn as nn


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors, use_xyz=True):
        super().__init__()
        self.use_xyz = True
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = farthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        if self.use_xyz:
            knn_x = torch.cat([knn_x, knn_xyz], dim=-1)
        # print(knn_x.shape)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, type, use_xyz=True):
        super().__init__()
        self.type = type
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.use_xyz = use_xyz

        if use_xyz:
            self.linear1 = Linear1Layer(3 + out_dim * expand, out_dim, bias=False)
        else:
            self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # lc_xyz [32,500,3] -> 每个point的坐标
        # knn_xyz [32,500,40,3] -> 每个point下采样的sample的坐标
        # Normalization
        if self.type == "mn40":
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            # mean_xyz = lc_xyz
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == "scan":
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G * K)).reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x, lc_xyz)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels / 2),
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(int(in_channels / 2)),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels / 2), out_channels=in_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class IGM(nn.Module):
    def __init__(self, group_num, k_neighbors, use_xyz=True):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors
        self.use_xyz = use_xyz

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        xyz_recptor = xyz[:, : N // 2, :].contiguous()
        xyz_ligrand = xyz[:, N // 2 :, :].contiguous()
        points_recptor = x[:, : N // 2, :].contiguous()
        points_ligrand = x[:, N // 2 :, :].contiguous()

        r_num = self.group_num // 2
        l_num = r_num

        # FPS

        fps_idx_r = farthest_point_sample(xyz_recptor, r_num).long()
        fps_idx_l = farthest_point_sample(xyz_ligrand, l_num).long()

        new_xyz_r = index_points(xyz_recptor, fps_idx_r)  # [B, r_num, 3]
        new_xyz_l = index_points(xyz_ligrand, fps_idx_l)  # [B, l_num, 3]
        new_points_r = index_points(points_recptor, fps_idx_r)  # [B, r_num, d]
        new_points_l = index_points(points_ligrand, fps_idx_l)  #

        lc_xyz = torch.cat([new_xyz_r, new_xyz_l], dim=1)
        lc_x = torch.cat([new_points_r, new_points_l], dim=1)
        # fps_idx = farthest_point_sample(xyz, self.group_num).long()
        # lc_xyz = index_points(xyz, fps_idx)
        # lc_x = index_points(x, fps_idx)

        # kNN

        idx_l = knn_point(self.k_neighbors, xyz_ligrand, new_xyz_r)
        grouped_xyz_r = index_points(xyz_ligrand, idx_l)  # [B, r_num, k, 3]
        grouped_points_r = index_points(points_ligrand, idx_l)  # [B, r_num, k, d]
        # ligrand anchor points group receptor points
        idx_r = knn_point(self.k_neighbors, xyz_recptor, new_xyz_l)
        grouped_xyz_l = index_points(xyz_recptor, idx_r)  # [B, l_num, k, 3]
        grouped_points_l = index_points(points_recptor, idx_r)

        if self.use_xyz:
            grouped_points_r = torch.cat([grouped_points_r, grouped_xyz_r], dim=-1)
            grouped_points_l = torch.cat([grouped_points_l, grouped_xyz_l], dim=-1)
        new_xyz = torch.cat([grouped_xyz_r, grouped_xyz_l], dim=1)  # [B, r_num+l_num, d]
        new_points = torch.cat([grouped_points_r, grouped_points_l], dim=1)
        # print(new_points.shape)
        return lc_xyz, lc_x, new_xyz, new_points


# PosE for Local Geometry Extraction
class _PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        device = knn_x.device
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // 2
        feat_range = torch.arange(feat_dim).float().to(device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        dist_xyz = torch.sqrt(torch.sum(knn_xyz**2, dim=1, keepdim=True))
        dist_xyz = torch.div(self.beta * dist_xyz.unsqueeze(-1), dim_embed)

        # div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(dist_xyz)
        cos_embed = torch.cos(dist_xyz)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)  # [bz,3,500,40]

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x, lc_xyz):
        device = knn_x.device
        B, _, G, K = knn_xyz.shape  # [batch_size,3,500,40]
        #  feat_dim = self.out_dim // (self.in_dim * 2)
        # lc_xyz [B,G,3]
        lc_xyz = lc_xyz.permute(0, 2, 1).contiguous().view(B, 3, G, 1)  # [batch_size,3,500,1]
        distances = (knn_xyz - lc_xyz.expand(-1, -1, -1, 40)).to(device)
        distances = torch.norm(distances, dim=1)
        sigma = 1.0
        weights = torch.exp(-distances / (2 * sigma**2))

        # Weigh
        # knn_x_w = knn_x + weights  # [batch_size,channel,500,40] + [batch_size,1,500,40]
        # knn_x_w *= weights

        knn_x_w = knn_x * weights.unsqueeze(1)

        return knn_x_w


# Parametric Encoder
class EncP(nn.Module):
    def __init__(
        self, in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type
    ):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.LGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points
        # Multi-stage Hierarchy

        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            if i == 0:
                self.FPS_kNN_list.append(IGM(group_num, k_neighbors))
            else:
                self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))

            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, LGA_block[i], dim_expansion[i], type))
            self.Pooling_list.append(Pooling(out_dim))

    def forward(self, x):
        xyz = x[:, :3, :]
        norm = x[:, 3:, :]
        xyz = xyz.permute(0, 2, 1)

        x = norm

        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


class SufrProNN(nn.Module):
    def __init__(
        self,
        in_channels=49,
        class_num=1,
        input_points=1000,
        num_stages=4,
        embed_dim=49,
        k_neighbors=40,
        beta=100,
        alpha=1000,
        LGA_block=[2, 1, 1, 1],
        dim_expansion=[2, 2, 2, 1],
        type="mn40",
    ):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(
            in_channels, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, LGA_block, dim_expansion, type
        )
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i

        self.out_channel += 2048

        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
        )

    def forward(self, x, plm):
        # xyz: point coordinates
        # x: point features

        x = self.EncP(x)  # 8 * 264

        x = torch.cat([x, plm], axis=1)  # [B,out_idm,N]    # 8*2312 * 1
        # temp = x
        x = self.classifier(x)

        x = torch.sigmoid(x)
        x = x.view(-1)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:2")
    xyz = torch.tensor(np.random.randint(0, 51, (32, 3, 1000)))
    point = torch.rand(32, 49, 1000)  # [B,D,N]  D = 29 + 7(atom_types) + 16(dmasif)
    point = torch.cat([xyz, point], axis=1).to(device)
    plm = torch.rand(32, 2048).to(device)
    print("===> testing SufrProNN ...")
    model = SufrProNN().to(device)

    print(
        "The model has {} millions parameterd".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
        )
    )

    out = model(point, plm)

    print(out.shape)
