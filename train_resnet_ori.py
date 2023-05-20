import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch_npu
from torch_npu.contrib import transfer_to_npu


# 定义结果保存文件夹
results_folder = "DXL/Results_ResNet"
models_folder = "DXL/Models"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)



# 定义数据预处理和增强的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载完整的数据集
full_dataset = ImageFolder("DXL/sound_Datasets", transform=transform)


# 划分训练集和测试集
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

test_indices = test_dataset.indices
test_labels = [full_dataset.targets[idx] for idx in test_indices]
print("test_labels", len(test_labels))

# 打印划分后的训练集和测试集大小
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))



# 按顺序打印类别标签
print("full_dataset.classes", full_dataset.classes)
classes = full_dataset.classes
# sorted_classes = sorted(classes)
sorted_classes = sorted(classes, key=lambda x: int(x))
print("classes", classes)
print("sorted_classes ", sorted_classes)



# 创建训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 在每10个epoch保存模型参数
def save_model(epoch):
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(models_folder, f'resnet_epoch_{epoch}.pth'))
        print(f"Model parameters saved for epoch {epoch}.")


# 在每个epoch结束后计算混淆矩阵并保存到文件
def save_confusion_matrix(cm, classes, epoch):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 排序类别标签
    sorted_classes = sorted(classes, key=lambda x: int(x))

    # 绘制混淆矩阵图像
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sorted_classes, yticklabels=sorted_classes)
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(results_folder, f'resnet_confusion_matrix_epoch_{epoch}.png'))
    plt.close()

    # 保存混淆矩阵文本文件
    np.savetxt(os.path.join(results_folder, f'resnet_confusion_matrix_epoch_{epoch}.txt'), cm_normalized, fmt='%.4f')
    print(f"Confusion matrix saved for epoch {epoch}.")

        
# 定义ResNet模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 12)  # 12分类任务，将全连接层的输出调整为12

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 将模型移到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

model = model.to(device)

# 训练模型
num_epochs = 80
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(predicted == labels.data)
        total += labels.size(0)

        # 每个batch打印当前的训练损失和准确率
        if (batch_idx + 1) % 10 == 0:
            batch_loss = running_loss / total
            batch_acc = 100.0 * running_corrects / total
            print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Train Loss: {batch_loss:.4f} - Train Acc: {batch_acc:.2f}%")

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * running_corrects / len(train_loader.dataset)


    # 验证阶段
    model.eval()
    val_corrects = 0
    val_total = 0
    val_predictions = []


    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # features = rearrange(images, 'b c h w -> b (h w) c')
            # outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

            val_corrects += torch.sum(predicted == labels.data)
            val_total += labels.size(0)

            val_predictions.extend(predicted.tolist())
            
#             print("val_predictions", val_predictions)
                
#             print("test_labels", test_labels)
            
            
#             # 计算混淆矩阵并保存到文件
#             cm = confusion_matrix(test_labels, val_predictions)
#             # cm = confusion_matrix(test_dataset.dataset.targets, val_predictions)
#             print("cm", cm)
            
#             save_confusion_matrix(cm, test_dataset.classes, epoch+1)

    val_acc = 100.0 * val_corrects / val_total


    # 计算混淆矩阵并保存到文件
    cm = confusion_matrix(test_labels, val_predictions)
    # cm = confusion_matrix(test_dataset.dataset.targets, val_predictions)
    save_confusion_matrix(cm, full_dataset.classes, epoch+1)

    # 输出每个epoch的训练损失、训练准确率和验证准确率
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    # 保存模型参数
    save_model(epoch)

    # 输出每个epoch的训练损失、训练准确率和验证准确率
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    
    
    # 训练完成后，你可以使用模型进行预测
