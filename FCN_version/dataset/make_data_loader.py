import sys
sys.path.append('/content/SiamCRNN/FCN_version')
import argparse
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import util_func.lovasz_loss as L
from PIL import Image

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader import OSCDDatset3Bands, make_data_loader, OSCDDatset13Bands
from util_func.metrics import Evaluator
from deep_networks.SiamCRNN import SiamCRNN

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.train_data_loader = make_data_loader(args)
        print(args.model_type + ' is running')
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = SiamCRNN(in_dim_1=3, in_dim_2=3)
        self.deep_model = self.deep_model.cuda()

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        self.deep_model.train()
        class_weight = torch.FloatTensor([1, 10]).cuda()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            self.optim.zero_grad()

            pre_img, post_img, bcd_labels, _ = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # bcd_output = self.deep_model(input_data)
            bcd_output = self.deep_model(pre_img, post_img)

            bcd_loss = F.cross_entropy(bcd_output, bcd_labels, weight=class_weight, ignore_index=255)
            lovasz_loss = L.lovasz_softmax(F.softmax(bcd_output, dim=1), bcd_labels, ignore=255)

            main_loss = bcd_loss + 0.75 * lovasz_loss
            main_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1},  change detection loss is {bcd_loss}'
                )
                if (itera + 1) % 200 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))

                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset_path = '/content/drive/MyDrive/Colab Notebooks/Levir-CD/testing'
        with open('/content/drive/MyDrive/Colab Notebooks/Levir-CD/test.txt', "r") as f:
            # data_name_list = f.read()
            data_name_list = [data_name.strip() for data_name in f]
        data_name_list = data_name_list
        dataset = OSCDDatset3Bands(dataset_path=dataset_path, data_list=data_name_list, crop_size=512,
                                   max_iters=None, type='test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=False)
        torch.cuda.empty_cache()

        for itera, data in enumerate(val_data_loader):
            pre_img, post_img, bcd_labels, data_name = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # bcd_output = self.deep_model(input_data)
            bcd_output = self.deep_model(pre_img, post_img)
            bcd_output = bcd_output.data.cpu().numpy()
            bcd_output = np.argmax(bcd_output, axis=1)

            bcd_img = bcd_output[0].copy()
            bcd_img[bcd_img == 1] = 255

            # imageio.imwrite('./' + data_name[0] + '.png', bcd_img)

            bcd_labels = bcd_labels.cpu().numpy()
            self.evaluator.add_batch(bcd_labels, bcd_output)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc

def main():
    parser = argparse.ArgumentParser(description="Training on OEM_OSM dataset")
    parser.add_argument('--dataset', type=str, default='OSCD_3Bands')
    parser.add_argument('--dataset_path', type=str,
                        default='/content/drive/MyDrive/Colab Notebooks/Levir-CD/training')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_data_list_path', type=str,
                        default='/content/drive/MyDrive/Colab Notebooks/Levir-CD/train.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--model_type', type=str, default='SiamCRNN')
    parser.add_argument('--model_param_path', type=str, default='/content/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list

    trainer = Trainer(args)
    trainer.training()

if __name__ == "__main__":
    main()

import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import dataset.imutils as imutils

band_idx = ['B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif',
            'B09.tif', 'B10.tif', 'B11.tif', 'B12.tif']

def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

def sentinel_loader(path):
    band_0_img = np.array(imageio.imread(os.path.join(path, 'B01.tif')), np.float32)
    ms_data = np.zeros((band_0_img.shape[0], band_0_img.shape[1], 3))
    for i, band in enumerate(band_idx):
        ms_data[:, :, i] = np.array(imageio.imread(os.path.join(path, band)), np.float32)

    return ms_data

def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the first position
    one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot

class OSCDDatset3Bands(Dataset):
    def __init__(self, dataset_path=None, data_list=None, crop_size=None, max_iters=None, type='train'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.crop_size = crop_size
        self.type = type
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_name = self.data_list[index]

        pre_img = img_loader(os.path.join(self.dataset_path, data_name, 'imgs_1_rect', 'img1.png'))
        post_img = img_loader(os.path.join(self.dataset_path, data_name, 'imgs_2_rect', 'img2.png'))
        bcd_label = img_loader(os.path.join(self.dataset_path, data_name, 'cm', 'cm.png'))

        # Resize pre_img and post_img to the specified crop size (256x256)
        pre_img = np.array(Image.fromarray(pre_img).resize((256, 256), Image.BICUBIC))
        post_img = np.array(Image.fromarray(post_img).resize((256, 256), Image.BICUBIC))

        if self.crop_size is not None:
            pre_img, post_img, bcd_label = imutils.random_crop(pre_img, post_img, bcd_label, self.crop_size)
        if self.type == 'train':
            pre_img, post_img, bcd_label = imutils.random_flip(pre_img, post_img, bcd_label)
            pre_img, post_img = imutils.random_rotate(pre_img, post_img)
        bcd_label[bcd_label == 255] = 1
        return pre_img.transpose((2, 0, 1)), post_img.transpose((2, 0, 1)), bcd_label, data_name

def make_data_loader(args):
    dataset = OSCDDatset3Bands(dataset_path=args.dataset_path, data_list=args.data_name_list, crop_size=args.crop_size,
                               max_iters=args.max_iters)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8,
                             pin_memory=True, drop_last=True)
    return data_loader
