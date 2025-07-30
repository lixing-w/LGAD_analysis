import torch
import torch.nn as nn 

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gru = nn.GRU(input_size=3, hidden_size=128, num_layers=2)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        # x = self.dropout_input(x) # shape: (N, 2), 2 for V, and Temperature
        out, _ = self.gru(x) # shape: (N, 8)
        out = self.fc(out) # shape: (N, 1), 1 for I
        return out

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
    def forward(self, x):
        B, N, _ = x.shape 
        out = self.mlp(x.reshape(B*N, 2))
        return out.reshape(B, N, 1)

class AutoEncoder(nn.Module):
    # first use autoencoder to learn compressed representations
    # of IV curve
    def __init__(self, max_seq_len):
        super().__init__()
        
        self.latent_dim = 16
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(16, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        
        dummy_input = torch.zeros(1, 1, max_seq_len)  # batch_size=1
        with torch.no_grad():
            self.cnn_out = self.encoder_cnn(dummy_input)
            self.flattened_dim = self.cnn_out.shape[1] * self.cnn_out.shape[2]

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, self.latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.flattened_dim),
            nn.LeakyReLU(),
            nn.Unflatten(1, (self.cnn_out.shape[1], self.cnn_out.shape[2])),
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 1, 3, stride=1, padding=1, output_padding=1),
        )
        
        # self.smooth_alpha = nn.Sequential( # dynamic denoising weight map
        #     nn.Conv1d(1, 16, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 3, 3, padding=1), 
        #     nn.Softmax(dim=1) # (B, 3, seq_len)
        # )
        
        # def gaussian_kernel1d(kernel_size, sigma):
        #     center = kernel_size // 2
        #     x = torch.arange(kernel_size) - center
        #     kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        #     kernel /= kernel.sum()
        #     return kernel

        # kernel_5 = gaussian_kernel1d(5, 1)
        # kernel_7 = gaussian_kernel1d(7, 1)
        # self.smooth_cnn_5 = nn.Conv1d(1, 1, 5, padding=2, bias=False)
        # self.smooth_cnn_7 = nn.Conv1d(1, 1, 7, padding=3, bias=False)
        # with torch.no_grad():
        #     self.smooth_cnn_5.weight.copy_(kernel_5.reshape(1, 1, -1))
        #     self.smooth_cnn_7.weight.copy_(kernel_7.reshape(1, 1, -1))
    
    def forward(self, x):
        features = self.encoder_cnn(x)
        latent_vec = self.encoder_fc(features)
        recons = self.decoder(latent_vec)
        # smooth_coeff = self.smooth_alpha(recons) # (B, 3, seq_len)
        # smoothed_recons = smooth_coeff[:,[0]] * recons + smooth_coeff[:,[1]] * self.smooth_cnn_5(recons) + smooth_coeff[:,[2]] * self.smooth_cnn_7(recons)
        return recons
        # return smoothed_recons, smooth_coeff[:,1:,:].mean(), torch.mean(torch.square(smoothed_recons - recons))