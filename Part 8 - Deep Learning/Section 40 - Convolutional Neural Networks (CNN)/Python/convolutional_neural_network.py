import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- SSL sertifika doğrulama sorununu geçici olarak devre dışı bırak ---
ssl._create_default_https_context = ssl._create_unverified_context

# Cihazı belirle (GPU kullanılabilir ise GPU, değilse CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Veriyi Hazırlama ve Dönüştürme
transform = transforms.Compose([
    transforms.ToTensor(),                               # Görseli [0,255] aralığından [0,1] aralığına çevir
    transforms.Normalize((0.1307,), (0.3081,))           # MNIST’in ortalama ve std ile normalize et
])

# MNIST veri setini indir ve yükle (SSL sorununu bypass ettik)
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# DataLoader ile batch halinde yükle
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)

# 2. Basit CNN Modeli Tanımı
def conv_block(in_channels, out_channels):
    """Helper fonksiyon: Conv -> ReLU -> MaxPool"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution
        nn.ReLU(inplace=True),                                          # Aktivasyon
        nn.MaxPool2d(kernel_size=2, stride=2)                           # 2x2 max pooling
    )

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # İki convolution bloğu
        self.block1 = conv_block(1, 32)    # Girdi kanal=1, çıktı kanal=32
        self.block2 = conv_block(32, 64)   # Girdi kanal=32, çıktı kanal=64
        # Fully connected katmanları
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flatten sonrası boyut
        self.fc2 = nn.Linear(128, 10)          # 10 çıktı (sınıf sayısı)

    def forward(self, x):
        # Convolution + Pooling adımları
        x = self.block1(x)  # 28x28 -> 14x14
        x = self.block2(x)  # 14x14 -> 7x7
        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 64*7*7]
        # Fully connected + ReLU
        x = F.relu(self.fc1(x))
        # Son katman (logits)
        x = self.fc2(x)
        return x

# Modeli oluştur ve cihaza taşı
model = SimpleCNN().to(device)

# 3. Kayıp Fonksiyonu ve Optimizasyon
criterion = nn.CrossEntropyLoss()             # Softmax + cross-entropy bir arada
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizasyon

# 4. Eğitim ve Test Fonksiyonları
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()                # Önceki gradyanları sıfırla
        outputs = model(data)                # İleri besleme (forward pass)
        loss = criterion(outputs, target)    # Kayıp değeri hesapla
        loss.backward()                      # Geri yayılım (backpropagation)
        optimizer.step()                     # Ağırlıkları güncelle
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

def evaluate():
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # Test aşamasında gradyan hesaplama kapatılır
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(target).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# 5. Eğitim Döngüsü
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch)
    evaluate()

print("Training complete!")