import os
import numpy as np
from .dataset import Transform

from .util import read_image

GHS_CLASSES = ["GHS01_Explosive",
               "GHS02_Flammable",
               "GHS03_Oxidizing",
               "GHS04_CommpressedGas",
               "GHS05_Corrosive",
               "GHS06_Toxic",
               "GHS07_Harmful",
               "GHS08_HealthHazard",
               "GHS09_EnviornmentalHazard"
               ]

class PicGHSDataSet:
    def __init__(self,
                 datafile_path,
                 min_imgsize,
                 max_imgsize,
                 test=False,
                 batch_size=None
                 ):
        self.ids = [id_.strip() for id_ in open(datafile_path)]
        self.label_names = GHS_CLASSES
        self.batch_size = batch_size
        self.test = test
        self.transformer = Transform(min_size=min_imgsize,
                                     max_size=max_imgsize,
                                     test=test
                                     )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        id_ = self.ids[idx]
        img, bboxes, labels = self.get_data_by_path(id_)
        img, ori_shape, bboxes, labels, scale = self.get_transformed_data(img, bboxes, labels)
        if self.test:
            difficult = [0 for _ in labels]
            return img.copy(), ori_shape, bboxes.copy(), labels.copy(), np.array(difficult)
        return img.copy(), bboxes.copy(), labels.copy(), scale
    
#    def __iter__(self):
#        if not self.batch_size:
#            for id_ in self.ids:
#                yield self.__getitem__(id_)
#        else:
#            imgs_batch = []
#            labels_batch = []
#            bboxes_batch = []
#            cnt = 0
#            for id_ in self.ids:
#                if cnt < self.batch_size:
#                    img, bbox, label = self.__getitem__(id_)
#                    imgs_batch.append(img)
#                    bboxes_batch.append(bbox)
#                    labels_batch.append(label)
#                    cnt += 1
#                else:
#                    yield imgs_batch, bboxes_batch, labels_batch
#                    imgs_batch = []
#                    labels_batch = []
#                    bboxes_batch = []
#                    cnt = 0
#            yield imgs_batch, bboxes_batch, labels_batch
    
    def get_data_by_path(self, imgpath):
        anno_file = imgpath.split('.')[0] + '.txt'
        bboxes = []
        labels = []
        img = read_image(imgpath, color=True)
        img_size = img.shape[1:]
        with open(anno_file, 'r') as f:
            for line in f:
                num, *box = line.split()
                label = int(num)
                bbox = [float(x) for x in box]
                bbox_t = self.translate_bbox(img_size, bbox)
                bboxes.append(bbox_t)
                labels.append(label)
        if len(bboxes) == 0:
            bboxes = [[]]
            labels = [[]]
        return img, np.array(bboxes), np.array(labels) 
    
    def get_transformed_data(self, ori_img, bbox, labels):
        img, ori_shape, bbox, label, scale = self.transformer((ori_img, bbox, labels))
        return img.copy(), ori_shape, bbox.copy(), label.copy(), scale

    @staticmethod
    def translate_bbox(img_size, bbox):
        height, width = img_size
        center_x, center_y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        x_max = int(((2 * center_x + bw) / 2) * width)
        x_min = int(((2 * center_x - bw) / 2) * width)
        y_min = int(((2 * center_y - bh) / 2) * height)
        y_max = int(((2 * center_y + bh) / 2) * height)
        return [y_min, x_min, y_max, x_max]
