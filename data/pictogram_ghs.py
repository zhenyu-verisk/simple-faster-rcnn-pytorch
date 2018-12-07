import os
import numpy as np

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
    def __init__(self, datafile_path, batch_size=None):
        self.ids = [id_.strip() for id_ in open(datafile_path)]
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        id_ = self.ids[idx]
        return self.get_data_by_path(id_)
    
    def __iter__(self):
        if not self.batch_size:
            for id_ in self.ids:
                yield self.get_data_by_path(id_)
        else:
            imgs_batch = []
            labels_batch = []
            bboxes_batch = []
            cnt = 0
            for id_ in self.ids:
                if cnt < self.batch_size:
                    img, bbox, label = self.get_data_by_path(id_)
                    imgs_batch.append(img)
                    bboxes_batch.append(bbox)
                    labels_batch.append(label)
                    cnt += 1
                else:
                    yield imgs_batch, bboxes_batch, labels_batch
                    imgs_batch = []
                    labels_batch = []
                    bboxes_batch = []
                    cnt = 0
    
    def get_data_by_path(self, imgpath):
        anno_file = imgpath.split('.')[0] + '.txt'
        bboxes = []
        labels = []
        img = read_image(imgpath, dtype=np.uint8, color=True)
        img_size = img.shape[1:]
        with open(anno_file, 'r') as f:
            for line in f:
                lable, **bbox = line.split()
                label = int(label)
                bbox = [float(x) for x in bbox]
                bbox_t = self.translate_bbox(img_size, bbox)
                bboxes.append(bbox_t)
                labels.append(label)
        return img, np.array(bboxes), np.array(label) 

    @classmethod
    def translate_bbox(cls, img_size, bbox):
        height, width = img_size
        center_x, center_y, bw, bh = bb[0], bb[1], bb[2], bb[3]
        x_max = int(((2 * center_x + bw) / 2) * width)
        x_min = int(((2 * center_x - bw) / 2) * width)
        y_min = int(((2 * center_y - bh) / 2) * height)
        y_max = int(((2 * center_y + bh) / 2) * height)
        return [y_min, x_min, y_max, x_max]
