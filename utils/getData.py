import cv2 as cv
import torch
import numpy as np
from os import listdir
from os.path import join
from torch.utils.data import Dataset

class getImageLabel(Dataset):
    def __init__ (self, 
                  augmented = './dataset/Augmented Images/Augmented Images/FOLDS_AUG/',
                  original = './dataset/Original Images/Original Images/FOLDS/',
                  folds = [1, 2, 3, 4, 5],):
        
        self.dataset = []
        to_one_hot = np.eye(6)

        for fold in folds:
            ori_train = join(original, f"fold{fold}/Train/")
            for i, pox in enumerate(sorted(listdir(ori_train))):
                for image_name in listdir(ori_train + "/" + pox):
                    image = cv.resize(cv.imread(ori_train + "/" + pox + "/" + image_name), (224, 224)) / 255
                    self.dataset.append([image, to_one_hot[i]])
            
            aug = join(augmented, f"fold{fold}_AUG/Train/")
            for i, pox in enumerate(sorted(listdir(aug))):
                for image_name in listdir(aug + "/" + pox):
                    image = cv.resize(cv.imread(aug + "/" + pox + "/" + image_name), (224, 224)) / 255
                    self.dataset.append([image, to_one_hot[i]])

            ori_test = join(original, f"fold{fold}/Test/")
            for i, pox in enumerate(sorted(listdir(ori_test))):
                for image_name in listdir(ori_test + "/" + pox):
                    image = cv.resize(cv.imread(ori_test + "/" + pox + "/" + image_name), (224, 224)) / 255
                    self.dataset.append([image, to_one_hot[i]])

            ori_valid = join(original, f"fold{fold}/Valid/")
            for i, pox in enumerate(sorted(listdir(ori_valid))):
                for image_name in listdir(ori_valid + "/" + pox):
                    image = cv.resize(cv.imread(ori_valid + "/" + pox + "/" + image_name), (224, 224)) / 255
                    self.dataset.append([image, to_one_hot[i]])
            
    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def __len__ (self):
        return len(self.dataset)
