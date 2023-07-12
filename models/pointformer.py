import sys

import torch
from timm.models.layers import trunc_normal_

sys.path.append("..")
sys.path.append("./")
from modules import *


# Hierarchical Encoder
class HEncoder(nn.Module):

    def __init__(self, encoder_depths=None, num_heads=6, encoder_dims=None,
                 local_radius=None):
        super().__init__()

        if local_radius is None:
            local_radius = [0.32, 0.64, 1.28]
        if encoder_dims is None:
            encoder_dims = [96, 192, 384]
        if encoder_depths is None:
            encoder_depths = [5, 5, 5]
        self.encoder_depths = encoder_depths
        self.encoder_num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(TokenEmbed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(TokenEmbed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))

            self.encoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.encoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
            ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(EncoderBlock(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=self.encoder_num_heads,
            ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs):
        # hierarchical encoding
        x_vis_list = []
        xyz_dist = None
        x_vis = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list


# finetune model
class PointSEG(nn.Module):
    def __init__(self, cls_dim=512):
        super().__init__()
        self.trans_dim = 384
        self.group_sizes = [16, 8, 8]
        self.num_groups = [512, 256, 64]
        self.cls_dim = cls_dim
        self.encoder_dims = [96, 192, 384]

        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = HEncoder()

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagations = nn.ModuleList()
        for i in range(3):
            self.propagations.append(
                PointNetFeaturePropagation_(in_channel=self.encoder_dims[i] + 3, mlp=[self.trans_dim * 4,
                                                                                      self.encoder_dims[i]]))
        out_dim = 0
        for x in self.encoder_dims:
            out_dim += x
        out_dim = 2 * out_dim
        self.conv = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, cls_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, is_eval=False):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2).contiguous()  # B N 3
        # divide the point cloud in the same form. This is important

        neighborhoods, centers, idxs = [], [], []
        center = None
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # b*g*k

        # hierarchical encoder
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs)
        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.propagations[i](pts.transpose(-1, -2), centers[i].transpose(-1, -2),
                                                 pts.transpose(-1, -2), x_vis_list[i])
        x = torch.cat((x_vis_list[0], x_vis_list[1], x_vis_list[2]), dim=1)  # 96 + 192 + 384
        x_max = torch.max(x, 2)[0]
        if is_eval:
            x_avg = torch.mean(x, 2)

            x = torch.cat((x_max, x_avg), 1)
            x = self.conv(x)
            x = F.log_softmax(x, dim=1)
            return x
        else:
            return x_max, x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss


if __name__ == '__main__':
    net = PointSEG()
    points = torch.rand(4, 3, 1024)
    x = net(points, True)
    print(x.shape)
