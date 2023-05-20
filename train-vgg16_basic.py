from __future__ import print_function, division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models

import time
import matplotlib as plt

import torch_npu
from torch_npu.contrib import transfer_to_npu

batch_size = 16
learning_rate = 0.0002
epoch = 200


dataset_name = "DXL/sound_Data_2"

os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# train_dir = './data/train'
# train_dir = 'F:/Image classification/structure dataset/train'
train_dir = 'DXL/sound_Data_2'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

# val_dir = './data/test'
# val_dir = 'F:/Image classification/structure dataset/test'
# val_dir = 'F:/code\Image classification/dataset6/test'
val_dir = 'DXL/sound_Data_2'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)


class VGGNet(nn.Module):
    def __init__(self, num_classes=12):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        # print("net",net)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print("x1111111",x.shape)
        x = x.view(x.size(0), -1)
        # print("x222222222222", x.shape)
        x = self.classifier(x)
        # print("x333333333333", x.shape)
        return x

# --------------------训练过程---------------------------------


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = VGGNet()
model = model.to(device)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []

for epoch in range(100):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:

        # print('new_batch_y', type(batch_y), batch_y.shape)

        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)

        # print('batch_x', batch_x.shape)
        # print('batch_y', batch_y, batch_y.shape)

        out = model(batch_x)
        # print("out", out.shape)
        # print("batch_y", batch_y.shape)
        loss = loss_func(out, batch_y)
        # train_loss += loss.data[0]
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        # train_acc += train_correct.data[0]
        train_acc += train_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))

    ##### Save model checkpoints #####
    if epoch % 20 == 0:
        torch.save(model.state_dict(),
                   "saved_models/%s/VGG16_expert_%d.pth" % (dataset_name, epoch))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x, batch_y = Variable(batch_x, volatile=True).to(device), Variable(batch_y, volatile=True).to(device)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # eval_loss += loss.data[0]
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        # eval_acc += num_correct.data[0]
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))


    Loss_list.append(eval_loss / (len(val_datasets)))
Accuracy_list.append(100 * eval_acc / (len(val_datasets)))



x1 = range(0, 100)
x2 = range(0, 100)
y1 = Accuracy_list
y2 = Loss_list
print("y1", y1)
print("y2", y2)
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.show()
# plt.savefig("F:/Image classification/PytorchExample-master/PytorchExample-master/data/accuracy_loss.jpg")
