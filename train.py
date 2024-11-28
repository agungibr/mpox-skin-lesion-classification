import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.getData import getImageLabel
from torch.optim import Adam
from torchvision import models
from torchvision.models import shufflenet_v2_x0_5
#from vit_pytorch import ViT

def main():
    BATCH_SIZE = 32
    EPOCH = 50
    LEARNING_RATE = 0.001
    folds = [1,2,3,4,5]
    device = 'cpu'

    for fold in folds:
        train_aug_loader = DataLoader(getImageLabel(augmented=f'./dataset/Augmented Images/Augmented Images/FOLDS_AUG/', folds=[fold], subdir='Train'), batch_size=BATCH_SIZE, shuffle=True)
        train_ori_loader = DataLoader(getImageLabel(original=f'./dataset/Original Images/Original Images/FOLDS/', folds=[fold], subdir='Train'), batch_size=BATCH_SIZE, shuffle=True)
        vali_loader = DataLoader(getImageLabel(original=f'./dataset/Original Images/Original Images/FOLDS/', folds=[fold], subdir='Valid'), batch_size=BATCH_SIZE, shuffle=False)

    #model = models.mobilenet_v3_small(pretrained=True)
    model = shufflenet_v2_x0_5(pretrained=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_vali_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        vali_loss = 0
        model.train()
        for batch, (src, trg) in enumerate(train_aug_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        for batch, (src, trg) in enumerate(train_ori_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        model.eval()
        for batch, (src, trg) in enumerate(vali_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            loss = loss_function(pred, trg)
            vali_loss += loss.detach().numpy()
            
        loss_train_all.append(train_loss / (len(train_aug_loader) + len(train_ori_loader)))
        loss_vali_all.append(vali_loss / len(vali_loader))
        print("epoch = ", epoch + 1, "train loss = ", train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            ", validation loss = ", vali_loss / len(vali_loader))
            
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            }, "./ShuffleNetV2_" + str(epoch + 1) + ".pt")
            
    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_vali_all, color="#3399e6", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./trainingShuffle50.png")

if __name__ == "__main__":
    main()