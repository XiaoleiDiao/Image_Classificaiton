import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from einops import rearrange
from einops.layers.torch import Rearrange

# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.fc(x)
        return x

# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.dataset = ImageFolder(data_path, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# 设置训练参数
image_size = 288
patch_size = 16
num_classes = 12
dim = 768
depth = 12
heads = 12
mlp_dim = 3072
batch_size = 32
num_epochs = 80
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])



# 加载完整的数据集
# full_dataset = ImageFolder("DXL/sound_Datasets", transform=transform)
full_dataset = ImageFolder("D:\Learn_Python\Image_classification\Degree_Segment_dataset", transform=transform)

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


# 创建训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# train_dataset = CustomDataset("DXL/sound_Datasets", transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = CustomDataset("DXL/sound_Datasets", transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 初始化ViT模型
model = ViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item()

    train_accuracy = 100 * train_correct / train_total

    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
