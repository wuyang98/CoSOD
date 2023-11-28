import os
from PIL import Image
from torch.utils import data
import transforms as trans
from torchvision import transforms
import random


def load_list(dataset_list, data_root):

    images = []

    dataset_list = dataset_list.split('+')

    for dataset_name in dataset_list:
        root = data_root + dataset_name + '/image/'
        for g_files in os.listdir(os.path.join(root)):
                names = os.listdir(os.path.join(root, g_files))
                images.append(list(map(lambda g_file: os.path.join(root, g_files, g_file), names)))
    return images


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, depth_transform, img_size, t_transform=None, label_14_transform=None,
                 label_28_transform=None, label_56_transform=None, label_112_transform=None, group_size=2):
        self.image_path = load_list(dataset_list, data_root)
        # print('image_path:', self.image_path)
        self.transform = transform
        self.depth_transform = depth_transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.img_size = img_size
        self.group = len(self.image_path)
        self.group_size = group_size
        self._len = 60000
        # self._len = 3

        # print(f'group:{self.group}')

    def __len__(self):
        return self._len
        
    def __getitem__(self, item):
        img_g = []
        label_g_14 = []
        label_g_28 = []
        label_g_56 = []
        label_g_112 = []
        label_g_224 = []
        contour_g_14 = []
        contour_g_28 = []
        contour_g_56 = []
        contour_g_112 = []
        contour_g_224 = []
        # print(self.group)
        if random.random() <= 0.05:
            # 78
            gitem = random.choice(range(30, self.group))
        else:
            gitem = random.choice(range(30))
        
        mitem = random.sample(range(len(self.image_path[gitem])), self.group_size)
        # print(f'group_size:{self.group_size}')
        for i in range(self.group_size):
            image = Image.open(self.image_path[gitem][mitem[i]]).convert('RGB')
            # label = Image.open(self.image_path[gitem][mitem[i]].replace('image', 'gt').replace('.jpg', '.png')).convert('L')
            # contour = Image.open(self.image_path[gitem][mitem[i]].replace('image', 'contour').replace('.jpg', '.png')).convert('L')
            label = Image.open(self.image_path[gitem][mitem[i]].replace('image', 'gt')).convert('L')
            contour = Image.open(self.image_path[gitem][mitem[i]].replace('image', 'contour')).convert('L')

            new_img = trans.Scale((224, 224))(image)
            new_label = trans.Scale((224, 224), interpolation=Image.NEAREST)(label)
            new_contour = trans.Scale((224, 224), interpolation=Image.NEAREST)(contour)

            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)

            img_g.append(new_img)
            label_g_14.append(label_14)
            label_g_28.append(label_28)
            label_g_56.append(label_56)
            label_g_112.append(label_112)
            label_g_224.append(label_224)
            contour_g_14.append(contour_14)
            contour_g_28.append(contour_28)
            contour_g_56.append(contour_56)
            contour_g_112.append(contour_112)
            contour_g_224.append(contour_224)
            
        return [img_g, label_g_224, label_g_14, label_g_28, label_g_56, label_g_112,\
                   contour_g_224, contour_g_14, contour_g_28, contour_g_56, contour_g_112]


def get_loader(dataset_list, data_root, img_size, group_size):

    transform = trans.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    depth_transform = trans.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t_transform = trans.Compose([
        transforms.ToTensor(),
    ])
    label_14_transform = trans.Compose([
        trans.Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    label_28_transform = trans.Compose([
        trans.Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    label_56_transform = trans.Compose([
        trans.Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    label_112_transform = trans.Compose([
        trans.Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = ImageData(dataset_list, data_root, transform, depth_transform, img_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform,group_size)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset