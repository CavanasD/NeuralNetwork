import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_epochs = 50 # 参数
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST data Loader
train_dataset = torchvision.datasets.MNIST(root="./data",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root="./data",train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
# CNN MODEL
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Test LOSS
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_losses = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return epoch_losses

# evaluate Func
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Plots
def plot_progress(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    # LOSS P
    plt.subplot(1, 2, 1)
    for i, losses in enumerate(train_losses):
        plt.plot(losses, label=f"Epoch {i+1}")
    plt.title("Training Loss per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    # ACC
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, marker="o", label="Accuracy")
    plt.title("Test Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.show()

# Show Prediction
def show_predictions(model, loader, device, num_images=50):
    model.eval()
    images_shown = 0
    rows, cols = 5, 10  # 一张图排布方式 5行10列
    plt.figure(figsize=(18, 10))

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(images)):
                if images_shown >= num_images:
                    break

                plt.subplot(rows, cols, images_shown + 1)
                plt.imshow(images[i].cpu().squeeze(), cmap="gray")

                color = "green" if predicted[i] == labels[i] else "red"
                plt.title(
                    f"pred: {predicted[i].item()}\ntrue: {labels[i].item()}",
                    color=color, fontsize=9
                )
                plt.axis("off")
                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()

# main Training Procedure
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    losses = train_one_epoch(model, train_loader, optimizer, device)
    train_losses.append(losses)

    acc = evaluate(model, test_loader, device)
    test_accuracies.append(acc)

    print(f"Epoch {epoch+1}: avg loss = {sum(losses)/len(losses):.4f}, test accuracy = {acc:.2f}%")

# Paint Loss&Acc Graph
plot_progress(train_losses, test_accuracies)

# Show Predict Result
show_predictions(model, test_loader, device, num_images=50)

# Save Model LOL
torch.save(model.state_dict(), "mnist_cnn50.ckpt")
