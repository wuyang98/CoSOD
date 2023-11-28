import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transforms as trans
from torchvision import transforms
from Models.ImageDepthNet_VQVAE import VQVAE


def train_vqvae(model: VQVAE, dataloader, device='cuda', ckpt_path='./checkpoints/vqvae/best_vqvae.pth',
             lr=1e-3, n_epochs=100, l_w_embedding=1, l_w_commitment=0.25):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    best_loss = 9999999999999999999.0
    for e in range(n_epochs):
        total_loss = 0

        for i, (x, name) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        if best_loss > total_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), ckpt_path)
        last_ckpt_path = ckpt_path.replace('best_vqvae.pth', 'last_vqvae.pth')
        torch.save(model.state_dict(), last_ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def load_list(data_root, dataset_name):

    images = []
    Names = []

    root = data_root + dataset_name +'/image/'
    for g_files in os.listdir(os.path.join(root)):
        names = os.listdir(os.path.join(root, g_files))
        for name in names:
            images.append(os.path.join(root, g_files, name))
            Names.append(name)

    return images, Names


class ImageData(Dataset):
    def __init__(self, data_root, dataset_name, transform, img_size, t_transform=None, group_size=2):
        self.image_path, self.img_names = load_list(data_root, dataset_name)
        self.transform = transform
        self.t_transform = t_transform
        self.img_size = img_size
        self.group = len(self.image_path)
        self.group_size = group_size
        self._len = 5

    def __len__(self):
        return self._len

    def __getitem__(self, item):

        image = Image.open(self.image_path[item]).convert('RGB')
        name = self.img_names[item]
        new_img = trans.Scale((224, 224))(image)

        new_img = self.transform(new_img)
        return new_img, name


def get_loader(data_root, dataset_name, img_size, group_size):

    transform = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    t_transform = trans.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageData(data_root, dataset_name, transform, img_size, t_transform, group_size)

    return dataset


if __name__ == '__main__':
    best_ckpt = '/home/dell/Codes/IJCV/IJCV2024/checkpoints/vqvae/best_vqvae.pth'
    train_dataset = get_loader(data_root='/home/dell/Codes/IJCV/data/', dataset_name='coco-seg', img_size=224, group_size=5)
    traindataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = VQVAE(input_dim=3, dim=384, n_embedding=128)
    train_vqvae(model, dataloader=traindataloader, ckpt_path=best_ckpt)
