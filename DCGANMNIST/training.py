import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
import torch.nn as nn
# Hyperparametros
batch_size = 128
epochs = 50
lr = 0.0002
beta1 = 0.5
nz = 100  # Tamaño del vector sonido 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargamos MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Inicializamos los modelos
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

# Optimizadores
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# TensorBoard
writer = SummaryWriter()

# Training Loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        # Update/Actualizamos discriminator
        netD.zero_grad()
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        output = netD(real_images)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Generate/Generamos   las imagenes "falsas"
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)
        label.fill_(0)
        output = netD(fake_images.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Actualizamos generador
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake_images)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output  stats y  imagenes para TensorBoard
        if i % 100 == 0:
            print(f'[{epoch}/{epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            writer.add_scalar('Loss/Discriminator', errD.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Generator', errG.item(), epoch * len(train_loader) + i)
            
            # Log images
            with torch.no_grad():
                fake_images = netG(noise).detach().cpu()
            img_grid_real = make_grid(real_images[:32], normalize=True)
            img_grid_fake = make_grid(fake_images[:32], normalize=True)
            writer.add_image('Real Images', img_grid_real, global_step=epoch * len(train_loader) + i)
            writer.add_image('Generated Images', img_grid_fake, global_step=epoch * len(train_loader) + i)

# Cerramos TensorBoard writer
writer.close()
