import cv2
import os

def Edge_Extract(root):
    image_category = os.listdir(root)
    for category in image_category:
        img_root = os.path.join(root, category)
        edge_root = os.path.join(root_edge, category)

        if not os.path.exists(edge_root):
            os.mkdir(edge_root)

        file_names = os.listdir(img_root)
        img_name = []

        for name in file_names:
            if not name.endswith('.jpg'):
                assert "This file %s is not JPG"%(name)
            img_name.append(os.path.join(img_root,name[:-4]+'.png'))

        index = 0
        for image in img_name:
            img = cv2.imread(image, 0)
            cv2.imwrite(edge_root+'/'+file_names[index], cv2.Canny(img, 30, 100))
            index += 1
        print(category)


if __name__ == '__main__':
    root = '/data/dataset/gt'
    root_edge = '/data/dataset/contour'
    Edge_Extract(root)
