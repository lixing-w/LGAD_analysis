import torch
import torch.nn as nn 

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
        
        self.latent_dim = 18
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(16, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        
        dummy_input = torch.zeros(1, 1, max_seq_len)  # batch_size=1
        with torch.no_grad():
            self.cnn_out = self.encoder_cnn(dummy_input)
            self.flattened_dim = self.cnn_out.shape[1] * self.cnn_out.shape[2]

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.flattened_dim),
            nn.Unflatten(1, (self.cnn_out.shape[1], self.cnn_out.shape[2])),
            nn.ConvTranspose1d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 1, 3, stride=1, padding=1, output_padding=0),
        )
        
        self.regressor = nn.Sequential( # regressor to map latent to performance metrics
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        # predicted metrics are 
        # [temp, date.toordinal(), humi, ramp_type, duration, sensor_number]
    
    def forward(self, x):
        features = self.encoder_cnn(x)
        latent_vec = self.encoder_fc(features)
        recons = self.decoder(latent_vec)
        metrics = self.regressor(latent_vec)
        return recons, metrics

class Encoder(AutoEncoder):
    
    def __init__(self, max_seq_len):
        super().__init__(max_seq_len)
        
    # just the encoder layers
    def forward(self, x):
        features = self.encoder_cnn(x)
        latent_vec = self.encoder_fc(features)
        return latent_vec

class Decoder(AutoEncoder):
    
    def __init__(self, max_seq_len):
        super().__init__(max_seq_len)
        
    # just the decoder layers
    def forward(self, x):
        recons = self.decoder(x)
        return recons 
    
class EnvToLatent(nn.Module):
    # an MLP that maps environmental conditions to latents
    def __init__(self):
        super().__init__()
        self.latent_dim = 18  # must match with AutoEncoder
        self.num_params = 6
        
        self.mlp = nn.Sequential(
            nn.Linear(self.num_params, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )
        
    def forward(self, x):
        p_latent = self.mlp(x)
        return p_latent