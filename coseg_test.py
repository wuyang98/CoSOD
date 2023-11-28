import os
from PIL import Image
import torch
from torchvision import transforms
from Models.ImageDepthNet_VQVAE import ImageDepthNet, VQVAE, PixelCNNWithEmbedding
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


def test(args, model_path, datapath, save_root_path, group_size, img_size, img_dir_name):

    vqvae = VQVAE(dim=args.n_dim, n_embedding=args.n_embedding)
    pixelcnn = PixelCNNWithEmbedding(n_blocks=15, p=256, linear_dim=args.linear_dim, bn=True, color_level=args.color_level)

    net = ImageDepthNet(args, vqvae, pixelcnn)
    net.cuda()
    net.eval()

    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k#[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    print(time.localtime(time.time()))
    with torch.no_grad():
        for p in range(len(datapath)):
            all_class = os.listdir(os.path.join(datapath[p], img_dir_name))
            image_list, save_list = list(), list()
            for s in range(len(all_class)):
                image_path = os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s]))
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                save_list.append(list(map(lambda x: os.path.join(save_root_path[p], all_class[s], x[:-4]+'.png'), image_path)))
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                if divided != 0:
                    for k in range(divided):
                        # k = 1
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_rgb = group_rgb.cuda()
                        outputs = net(group_rgb, group_rgb)
                        outputs_saliency, _ = outputs
                        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                        output_s_1 = F.sigmoid(mask_1_1)
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = output_s_1.squeeze()
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size - rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.cuda()
                    outputs = net(group_rgb, group_rgb)
                    outputs_saliency, _ = outputs
                    mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                    output_s_1 = F.sigmoid(mask_1_1)
                    cur_class_mask[(divided * group_size):] = output_s_1[:rested].squeeze()

                class_save_path = os.path.join(save_root_path[p], all_class[i])
                if not os.path.exists(class_save_path):
                    os.makedirs(class_save_path)

                for j in range(len(cur_class_all_image)):
                    exact_save_path = save_list[i][j]
                    result = cur_class_mask[j, :, :].numpy()
                    result = Image.fromarray(result * 255)
                    w, h = Image.open(image_list[i][j]).size
                    result = result.resize((w, h), Image.BILINEAR)
                    result.convert('L').save(exact_save_path)

            print('done')
    print(time.localtime(time.time()))


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    model_path = "./result/models/baseline_last.pth"
    
    # val_datapath = ["/data/chook_test/"]
    # save_root_path = ['./result/test/chook']
    val_datapath = ["/home/dell/Codes/IJCV/data/CoCA"]
    save_root_path = ['./result/rebuttal/CoCA']
    import argparse

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='/data/', type=str, help='data path')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--group_size', default=5,  type=int, help='network input size')
    parser.add_argument('--mode', default='test',  type=str, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str,
                        help='load pretrained model')
    # parser.add_argument('')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--n_embedding', type=int, default=128, help='path for saving result.txt')
    parser.add_argument('--n_dim', type=int, default=384, help='path for saving result.txt')
    parser.add_argument('--color_level', type=int, default=128, help='path for saving result.txt')
    parser.add_argument('--linear_dim', type=int, default=128, help='path for saving result.txt')

    args = parser.parse_args()

    test(args, model_path, val_datapath, save_root_path, 5, 224, 'image',)
