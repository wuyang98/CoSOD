import os
import torch
import Training
from Evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33113', type=str, help='init_method')
    parser.add_argument('--data_root', default='/home/hsl/Co-Saliency/data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--group_size', default=5, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=20000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=40000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--val_iter', default=1000, type=int, help='for val')
    parser.add_argument('--show_iter', default=100, type=int, help='for show in txt')
    parser.add_argument('--trainset', default='coco_78+DUTS_class_or+sub', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--mode', default='train', type=str, help='mode')
    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='cosal2015')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGBD_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    parser.add_argument('--save_vqvae', type=str, default='./checkpoints/VQVAE/', help='path for saving result.txt')
    parser.add_argument('--save_gen_model', type=str, default='./checkpoints/VQVAE/',
                        help='path for saving result.txt')

    parser.add_argument('--n_embedding', type=int, default=128, help='path for saving result.txt')
    parser.add_argument('--n_dim', type=int, default=384, help='path for saving result.txt')
    parser.add_argument('--color_level', type=int, default=128, help='path for saving result.txt')
    parser.add_argument('--linear_dim', type=int, default=128, help='path for saving result.txt')

    parser.add_argument('--gen_model_epoch', type=int, default=50, help='path for saving result.txt')
    parser.add_argument('--vqvae_epoch', type=int, default=100, help='path for saving result.txt')
    args = parser.parse_args()
    torch.cuda.current_device()
    torch.cuda._initialized = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # num_gpus = torch.cuda.device_count()
    # print('+++++++++++++')
    # print(num_gpus)
    # torch.cuda.set_device(2)
    num_gpus = 1
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
