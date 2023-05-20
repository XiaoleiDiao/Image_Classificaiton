import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch_npu
from torch_npu.contrib import transfer_to_npu

# 定义结果保存文件夹
results_folder = "DXL/Results"
os.makedirs(results_folder, exist_ok=True)



# 定义计算混淆矩阵并保存到文件
def save_confusion_matrix(cm, classes):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 排序类别标签
    sorted_classes = sorted(classes, key=lambda x: int(x))

    # 绘制混淆矩阵图像
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sorted_classes, yticklabels=sorted_classes)
    plt.title(f"Confusion Matrix (ResNet)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(results_folder, f'resnet_confusion_matrix_resnet_epoch_49_1.png'))
    plt.close()

    # 保存混淆矩阵文本文件
    np.savetxt(os.path.join(results_folder, f'resnet_confusion_matrix_resnet_epoch_49_1.txt'), cm_normalized, fmt='%.4f')
    print(f"Confusion matrix saved for resnet_epoch_49_1.pth.")


# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试集数据
test_dataset = ImageFolder("DXL/sound_Datasets", transform=transform)

# 创建测试集的数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义ResNet模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 12)  # 假设有12个类别

# 加载训练好的模型参数
model.load_state_dict(torch.load("DXL/Models/resnet_epoch_49_1.pth"))

# 将模型移到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 测试阶段
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        predictions.extend(predicted.tolist())
        targets.extend(labels.tolist())

# 计算分类准确率
accuracy = accuracy_score(targets, predictions)
print(f"Accuracy: {accuracy:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(targets, predictions)

print("Confusion Matrix:")
print(cm)

# 计算混淆矩阵并保存到文件
# cm = confusion_matrix(test_labels, val_predictions)
save_confusion_matrix(cm, test_dataset.classes)


# # 保存混淆矩阵文本文件
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# np.savetxt(os.path.join(results_folder, 'confusion_matrix.txt'), cm_normalized, fmt='%.2f')

# # 保存混淆矩阵准确率图像
# plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.colorbar()
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
# plt.show()


# # 保存混淆矩阵文本文件
# np.savetxt(os.path.join(results_folder, 'ResNet_test_confusion_matrix.txt'), cm, fmt='%d')


# # 绘制混淆矩阵图
# plt.figure(figsize=(12, 10))
# classes = test_dataset.classes
# sns.heatmap(cm, annot=True, fmt=".2%", cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.savefig(os.path.join(results_folder, 'ResNet_test_confusion_matrix.png'))
