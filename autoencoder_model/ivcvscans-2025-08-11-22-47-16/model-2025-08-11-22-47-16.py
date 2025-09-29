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
            nn.Conv1d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([32, max_seq_len]),
            nn.Conv1d(32, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([128, max_seq_len // 2]),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([128, max_seq_len // 4]),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
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
            nn.LayerNorm(128),
            nn.Linear(128, self.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.flattened_dim),
            nn.Unflatten(1, (self.cnn_out.shape[1], self.cnn_out.shape[2])),
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([128, max_seq_len // 4]),
            nn.ConvTranspose1d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([128, max_seq_len // 2]),
            nn.ConvTranspose1d(128, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 2, 3, stride=1, padding=1, output_padding=0), # 2 output channels, 1 for current, 1 for end_prob
        )
        
        self.regressor = nn.Sequential( # regressor to map latent to performance metrics
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
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

class VAE(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        
        self.latent_dim = 80
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([32, max_seq_len]),
            nn.Conv1d(32, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([256, max_seq_len // 2]),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([256, max_seq_len // 4]),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        
        dummy_input = torch.zeros(1, 1, max_seq_len)  # batch_size=1
        with torch.no_grad():
            self.cnn_out = self.encoder_cnn(dummy_input)
            self.flattened_dim = self.cnn_out.shape[1] * self.cnn_out.shape[2]

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 256),
            nn.LeakyReLU(),
        )
        
        # self.embed = nn.Linear(1, 64)  
        # self.pos = nn.Parameter(torch.randn(1, max_seq_len, 64))
        # encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.mu_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.latent_dim), # predicts mu of p(z|x)
        )
        self.logvar_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, self.latent_dim) # predicts log(std^2) of p(z|x)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.flattened_dim),
            nn.Unflatten(1, (self.cnn_out.shape[1], self.cnn_out.shape[2])),
            nn.ConvTranspose1d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([256, max_seq_len // 4]),
            nn.ConvTranspose1d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([256, max_seq_len // 2]),
            nn.ConvTranspose1d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 2, 3, stride=1, padding=1, output_padding=0), # 2 output channels, 1 for current, 1 for end_prob
        )

    def reparamatrize(self, mu, logvar): # samples a latent vector from the given distribution
        std = torch.exp(0.5 * logvar) # convert to std
        eps = torch.randn_like(std)
        z = mu + eps * std 
        return z 
    
    def forward(self, x):
        cnn_features = self.encoder_cnn(x)
        features = self.encoder_fc(cnn_features)
        # x = x.permute(0, 2, 1)
        # x = self.embed(x) + self.pos
        # out = self.transformer(x)
        # features = out.mean(dim=1)
        mu = self.mu_fc(features)
        logvar = self.logvar_fc(features)
        
        # sample a z
        latent_vec = self.reparamatrize(mu, logvar)
        recons = self.decoder(latent_vec)
        if self.training:
            return recons, mu, logvar 
        else:
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
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.LayerNorm(256),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )
        
    def forward(self, x):
        p_latent_and_seq_len = self.mlp(x)
        return p_latent_and_seq_len