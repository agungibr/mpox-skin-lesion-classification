import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.getData import getImageLabel
from torchvision import models
from torchvision.models import shufflenet_v2_x0_5
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    folds = [1,2,3,4,5]
    device = 'cpu'

    for fold in folds:
        test_loader = DataLoader(getImageLabel(original=f'./dataset/Original Images/Original Images/FOLDS/', folds=[fold], subdir='Test'), batch_size=BATCH_SIZE, shuffle=True)

    #model = models.mobilenet_v3_small(pretrained=True)
    model = shufflenet_v2_x0_5(pretrained=True)
    #model = models.squeezenet1_0(pretrained=True)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load('ShuffleNetV2_50.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_batch = checkpoint['loss']

    prediction, ground_truth = [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            prediction.extend(torch.argmax(pred,dim=1).detach().numpy())
            #ground_truth.extend(torch.argmax(trg, dim=1).detach().numpy())
            ground_truth.extend(trg.detach().numpy())

    classes = ('Chikenpox', 'Cowpox', 'Healty', 'HFMD', 'Measles', 'Monkeypox')

    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')

    print("accuracy score = ", accuracy_score(ground_truth, prediction))
    print("precision score = ", precision_score(ground_truth, prediction, average='weighted'))
    print("recall score = ", recall_score(ground_truth, prediction, average='weighted'))
    print("f1 score score = ", f1_score(ground_truth, prediction, average='weighted'))

if __name__ == "__main__":
    main()
