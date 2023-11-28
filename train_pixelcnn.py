import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transforms as trans
from torchvision import transforms
from Models.ImageDepthNet_VQVAE import VQVAE, PixelCNNWithEmbedding

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


def train_generative_model(vqvae: VQVAE, model, dataloader, device='cuda', ckpt_path='dldemos/VQVAE/gen_model.pth', n_epochs=50):

    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    best_loss = 9999999999999999999999.0
    for e in range(n_epochs):
        total_loss = 0
        for i, (x, name) in enumerate(dataloader):
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)
            predict_x = model(x)

            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        if best_loss > total_loss:
            torch.save(model.state_dict(), ckpt_path)
        last_ckpt = ckpt_path.replace('best_gen.pth', 'last_gen.pth')
        torch.save(model.state_dict(), last_ckpt)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


if __name__ == "__main__":
    device = 'cuda'
    best_ckpt = '/home/dell/Codes/IJCV/IJCV2024/checkpoints/vqvae/best_gen.pth'
    vqvae_para = '/home/dell/Codes/IJCV/IJCV2024/checkpoints/vqvae/best_vqvae.pth'
    vqvae = VQVAE(input_dim=3, dim=384, n_embedding=128)
    vqvae.load_state_dict(torch.load(vqvae_para))
    gen_model = PixelCNNWithEmbedding(n_blocks=15, p=256, linear_dim=128, bn=True, color_level=128)
    train_dataset = get_loader(data_root='/home/dell/Codes/IJCV/data/', dataset_name='coco-seg', img_size=224, group_size=5)
    traindataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_generative_model(vqvae, gen_model, traindataloader, device=device, ckpt_path=best_ckpt, n_epochs=50)