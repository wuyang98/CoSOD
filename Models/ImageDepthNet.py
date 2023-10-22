import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torch
from .pixelcnn import GatedBlock, GatedPixelCNN
import matplotlib.pyplot as plt
import torch.nn.functional as F


def b2g(feature, B, G):
    out = torch.chunk(feature, B, dim=0)
    return out


def g2b(feature, b,g):
    out = torch.cat([torch.mean(feat, keepdim=True, dim=0) for feat in torch.chunk(feature, g, dim=0)], dim=0)
    return out


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAE(nn.Module):

    def __init__(self, input_dim=3, dim=128, n_embedding=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1),
        #     ResidualBlock(dim), ResidualBlock(dim),
        #     nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
        #     nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.decoder1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
        )
        self.ConvT1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU())
        self.ConvT2 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU())
        self.ConvT3 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU())
        self.ConvT4 = nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)

        # yaogai
        self.n_downsample = 4

    def forward(self, x):
        # encode
        # print(f'X:{x.shape}')
        ze = self.encoder(x)

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        # print(f'embedding:{embedding}')
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat_m = self.decoder1(decoder_input)
        ConvT1 = self.ConvT1(x_hat_m)
        ConvT2 = self.ConvT2(ConvT1)
        COnvT3 = self.ConvT3(ConvT2)
        x_hat = self.ConvT4(COnvT3)
        # x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat_m = self.decoder1(zq)
        ConvT1 = self.ConvT1(x_hat_m)
        ConvT2 = self.ConvT2(ConvT1)
        COnvT3 = self.ConvT3(ConvT2)
        x_hat = self.ConvT4(COnvT3)
        # x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)


class PixelCNNWithEmbedding(GatedPixelCNN):

    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level=256):
        super().__init__(n_blocks, p, linear_dim, bn, color_level)
        self.embedding = nn.Embedding(color_level, p)
        self.block1 = GatedBlock('A', p, p, bn)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return super().forward(x)

class ImageDepthNet(nn.Module):
    def __init__(self, args, vqvae, gen_model):
        super(ImageDepthNet, self).__init__()

        self.vqvae = vqvae
        self.gen_model = gen_model
        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        # self.VQVAE = VQVAE()
        # self.PixelCNN = PixelCNNWithEmbedding(n_blocks=15, p=384, linear_dim=256, bn=True, color_level=32)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        
        self.group_size = args.group_size

        self.fc1 = nn.Linear(8 * 11 * 11, 8)
        self.fc2 = nn.Linear(8 * 11 * 11, 8)
        self.fc3 = nn.Linear(3, 1)
        # self.xy_encoder = Encoder_xy(5, 4, 3)
        # self.x_encoder = Encoder_x(5, 3, 3)
        self.spatial_axes = [1, 2]

        self.mode = args.mode
        # self.decode = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1)
        # )

    def loss(self, pred_x, x):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(pred_x, x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        return torch.index_select(a, dim, order_index)


    def forward(self, image_Input, label):
        B, C, H, W = image_Input.shape
        saliency_fea_1_16 = []
        fea_1_16 = []
        saliency_tokens = []
        contour_fea_1_16 = []
        contour_tokens = []
        co_tokens = []
        if self.mode == 'test':
            self.group_size = image_Input.shape[0]

        self.batch_size = image_Input.shape[0]//self.group_size
        # CoSOD Transformer Branch
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        q_repeats = b2g(rgb_fea_1_16, self.batch_size, self.group_size)

        # VQVAE
        if self.batch_size == 2:
            z = []
            H, W = self.vqvae.get_latent_HW((C, H, W))
            feature1 = self.vqvae.encode(image_Input[0:5, :, :, :])
            feature2 = self.vqvae.encode(image_Input[5:, :, :, :])
            feature = torch.cat((feature1, feature2), dim=0)

            x = torch.zeros((B, H, W)).cuda().to(torch.long)
            for i in range(H):
                for j in range(W):
                    predict = self.gen_model(feature)
                    prob_dist = F.softmax(predict[:, :, i, j], -1)
                    pixel = torch.multinomial(prob_dist, 1)
                    x[:, i, j] = pixel[:, 0]
            img = self.vqvae.vq_embedding(x).permute(0, 3, 1, 2)
            img1 = img[0:5, :, :, :]
            img2 = img[5:, :, :, :]
            z1 = self.vqvae.decoder1(img1)
            z2 = self.vqvae.decoder1(img2)
            z.append(z1)
            z.append(z2)
        else:
            z = []
            # print('batch size is not 2')
            H, W = self.vqvae.get_latent_HW((C, H, W))
            feature = self.vqvae.encode(image_Input[0:5, :, :, :])
            x = torch.zeros((B, H, W)).cuda().to(torch.long)
            with torch.no_grad():
                for i in range(H):
                    for j in range(W):
                        predict = self.gen_model(feature)
                        prob_dist = F.softmax(predict[:, :, i, j], -1)
                        pixel = torch.multinomial(prob_dist, 1)
                        x[:, i, j] = pixel[:, 0]
            img = self.vqvae.vq_embedding(x).permute(0, 3, 1, 2)
            z1 = self.vqvae.decoder1(img)
            z.append(z1)
        # VST Convertor
        aa = 0
        for q in q_repeats:
            # if self.batch_size == 2:
            qk = q.reshape(1, -1, 384)  # 1*980*384
            w = z[aa]
            wk = w.reshape(1, -1, 384)  # 1*980*384
            aa += 1
            memo, neno, co_token = self.transformer(qk, wk)
            q = memo.reshape(self.group_size, -1, 384)  # 5*196*384
            w = neno.reshape(self.group_size, -1, 384)  # 5*196*384
            mask = self.token_trans(q, w)
            saliency_fea_1_16.append(mask[0])
            fea_1_16.append(mask[1])
            saliency_tokens.append(mask[2])
            contour_fea_1_16.append(mask[3])
            contour_tokens.append(mask[4])
            co_tokens.append(co_token.repeat(self.group_size, 1, 1))

        saliency_fea_1_16 = torch.cat(saliency_fea_1_16, dim=0)
        fea_1_16 = torch.cat(fea_1_16, dim=0)
        saliency_tokens = torch.cat(saliency_tokens, dim=0)
        contour_fea_1_16 = torch.cat(contour_fea_1_16, dim=0)
        contour_tokens = torch.cat(contour_tokens, dim=0)
        co_tokens = torch.cat(co_tokens, dim=0)

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4,co_tokens)

        return outputs
