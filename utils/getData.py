import cv2 as cv
import torch
import numpy as np
from os import listdir
from torch.utils.data import Dataset

class getImageLabel(Dataset):
    def __init__ (self, 
                  augmented = None,
                  original = None,
                  folds = [1, 2, 3, 4, 5],
                  subdir = ['Train', 'Test', 'Valid']):
        
        self.dataset = []
        to_one_hot = np.eye(6)

        for fold in folds:
            if original:
                path = (original + f"fold{fold}/{subdir}")
                for i, pox in enumerate(sorted(listdir(path))):
                    for image_name in listdir(path + "/" + pox):
                        image = cv.resize(cv.imread(path + "/" + pox + "/" + image_name), (32, 32)) / 255
                        self.dataset.append([image, to_one_hot[i]])

            if augmented and subdir == 'Train':
                path = (augmented + f"fold{fold}_AUG/Train")
                for i, pox in enumerate(sorted(listdir(path))):
                    for image_name in listdir(path + "/" + pox):
                        image = cv.resize(cv.imread(path + "/" + pox + "/" + image_name), (32, 32)) / 255
                        self.dataset.append([image, to_one_hot[i]])
            
    def __getitem__(self, item):
        feature, label = self.dataset[item]
        #return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        label_index = torch.argmax(torch.tensor(label, dtype=torch.float32))
        return torch.tensor(feature, dtype=torch.float32), label_index
    
    def __len__ (self):
        return len(self.dataset)