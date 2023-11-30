# Towards Open-World Co-Salient Object Detection with Generative Uncertainty-aware Group Selective Exchange-Masking

Thank you very much for your interest in our work.

The OWCoSOD datasets (OWCoSal, OWCoSOD, OWCoCA) can be downloaded from the link https://pan.baidu.com/s/11MKqPIRP58p8lvz7x9AF2Q, and the password is 1310.

The results of our method on OWCoSal, OWCoSOD, OWCoCA can be downloaded from https://pan.baidu.com/s/1Yw3jN_cxkRgR47PSiIclPw, the password is 1310.

# Co-Salient Object Detection with Uncertainty-aware Group Exchange-Masking

The results of our method on CoSal2015, CoSOD3k, CoCA, MSRC, and iCoseg are available, and they can be downloaded from the link: https://pan.baidu.com/s/1uRwH5Y1HgDvxWd9gwRoR9g, and the password is 1310.

The pretrained_model and weights can be downloaded from the link: https://pan.baidu.com/s/1_FVoR6QP6FeQCZEGxgyZog and the password is 1310.

Putting the pretrained_model into ./pretrained_model and weights files into ./result/models and run coseg_test.py can get the results.

The link of the eval toolbox is: https://github.com/zzhanghub/eval-co-sod, we are very grateful for their contributions.

### Installation Instructions
    - We use Python 3.8
    - Pytorch 2.1.1(CUDA 12.1 build).
Please see `requirements.txt` for all the other requirments.

### Train on coco-seg
When we initialy train out method, we need firstly train vqvae and pixelcnn

Train vqvae

    python train_VQVAE.py
Train pixelcnn

    python train_pixelcnn.py
Train our Method

    python main.py --data_root /home/dell/Codes/IJCV/data/  --trainset coco-seg --n_embedding 128 --n_dim 384 --color_level 128 --linear_dim 128 --save_vqvae ./checkpoints/vqvae --save_gen_model ./checkpoints/vqvae
### Test on CoCA, CoSOD3k, CoSal2015
Test on CoCA

    python coseg_test.py --n_embedding 128 --n_dim 384 --color_level 128 --linear_dim 128

### PS
