import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

transform = transforms.ToTensor()
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])
minst_data = datasets.MNIST(root='', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=minst_data, batch_size=64, shuffle=True)

dataiter = iter(dataloader)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))
print(images.shape)
# print(images[0])

'''for i in range(3):
    plt.imshow(images[i].view(28,28))
    print(labels[i])
    plt.show()'''


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # [0, 1]
        )  # note if [-1, 1] --> nn.Tanh

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Convo_Autoencoder(nn.Module):
    def __init__(self):
        # N, 1, 28, 28
        super(Convo_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),      # N, 16, 14, 14 --> [(28-3)/2 + 1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),     # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)            # N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),      # N, 32, 7, 7 --> [ s(1-1) +7 -2P]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),    # N, 16, 2(7-1) +3 -2(1) + out
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvoMax_Autoencoder(nn.Module):
    def __init__(self):
        # N, 1, 28, 28
        super(ConvoMax_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),      # N, 16, 14, 14 --> [(28-3)/2 + 1
            nn.ReLU(),
            nn.MaxPool2d(2),              # N, 32, 7, 7
            nn.Conv2d(32, 64, 7)            # N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),      # N, 32, 7, 7 --> [ s(1-1) +7 -2P]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),    # N, 16, 2(7-1) +3 -2(1) + out
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available')

# model = Autoencoder().to(device)          # Linear
model = Convo_Autoencoder().to(device)
# model = ConvoMax_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# training
epoch_iter = 10
outputs = []
for epoch in range(epoch_iter):
    for (image, _) in tqdm(dataloader):
        image = image.to(device)
        # image = image.view(-1, 28 * 28)
        output = model(image)
        loss = criterion(output, image)  # autoencoder --> back to original picture

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch {}: loss {:.4f}'.format(epoch + 1, loss.item()))
    image, output = image.to('cpu'), output.to('cpu')
    outputs.append((epoch, image, output))

# image show
for k in range(0, epoch_iter, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()

    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)       # row_length + i + 1
        # item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
plt.show()
