import torch.nn as nn

class ScalarEmbedder(nn.Module):
    def __init__(self):
        super(ScalarEmbedder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), 
            nn.ReLU(),
            nn.Linear(16, 1)   
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 16),   
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),   
            nn.Sigmoid()        
        )

    def forward(self, x):
        # Forward pass: encode the input to a scalar, then decode it
        scalar_embedding = self.encoder(x)
        reconstructed = self.decoder(scalar_embedding)
        return reconstructed, scalar_embedding

