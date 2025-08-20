import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            # Input: [3, 96, 96]
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 48, 48]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32, 24, 24]

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 12, 12]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),  # -> [64, 6, 6]

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> [128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc_enc = nn.Linear(128 * 3 * 3, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 256)
        self.fc_dec2 = nn.Linear(256, 128 * 3 * 3)
        self.decoder = nn.Sequential(
            # Start with tensor of shape [128, 3, 3]
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [128, 6, 6]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [64, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [32, 24, 24]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [32, 48, 48]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [16, 96, 96]
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # -> [3, 96, 96]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc_enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_dec1(z))
        h = torch.relu(self.fc_dec2(h))
        h = h.view(h.size(0), 128, 3, 3)
        x_recon = self.decoder(h)
        return torch.sigmoid(x_recon)  # assume image pixel range [0,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar