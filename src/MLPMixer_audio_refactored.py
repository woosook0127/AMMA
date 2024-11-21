import torch
import torch.nn as nn
import torch.nn.functional as F

class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, spectrograms):
        batch_size = spectrograms.size(0)
        # Extract patches using unfold for spectrograms
        patches = spectrograms.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * spectrograms.size(1))
        return patches


class MLPBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim):
        super(MLPBlock, self).__init__()
        
        # Token Mixing: Apply a transformation along the patch dimension
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # Normalize across the channel dimension
            nn.Linear(num_patches, token_mixing_dim),
            nn.GELU(),
            nn.Linear(token_mixing_dim, num_patches)
        )
        
        # Channel Mixing: Apply a transformation along the channel dimension
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # Normalize across the channel dimension
            nn.Linear(hidden_dim, channel_mixing_dim),
            nn.GELU(),
            nn.Linear(channel_mixing_dim, hidden_dim)
        )

    def forward(self, X):
        # X shape: (batch_size, num_patches, hidden_dim)
        
        # Token Mixing: Transpose and apply the mixing along the patch dimension
        X_T = X.transpose(1, 2)  # (batch_size, hidden_dim, num_patches)
        X_T = self.token_mixing(X_T)
        X_prime = X_T.transpose(1, 2)  # Transpose back to (batch_size, num_patches, hidden_dim)
        U = X + X_prime  # Add skip connection
        
        # Channel Mixing: Apply mixing along the channel dimension
        X_C = self.channel_mixing(U)
        Y = U + X_C  # Add skip connection
        
        return Y


class MLPMixer(nn.Module):
    def __init__(self, patch_size, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim, num_of_mlp_blocks, freq_bins, time_frames, num_classes):
        super(MLPMixer, self).__init__()
        self.projection = nn.Linear(patch_size * patch_size, hidden_dim)
        self.mlp_blocks = nn.ModuleList([MLPBlock(num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim) for _ in range(num_of_mlp_blocks)])

        self.data_augmentation = nn.Sequential(
            nn.Dropout(0.1),
        )

        self.classification_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.patches = Patches(patch_size)

    def forward(self, spectrograms):
        batch_size = spectrograms.size(0)

        # Data augmentation
        augmented_spectrograms = self.data_augmentation(spectrograms)

        # Extract patches from spectrograms
        X = self.patches(augmented_spectrograms)

        # Per-patch Fully-connected
        X = self.projection(X)

        # MLP Blocks
        for block in self.mlp_blocks:
            X = block(X)

        # Global average pooling across patches
        X = X.mean(dim=1)  # (batch_size, hidden_dim)

        # Classification layer
        out = self.classification_layer(X)

        return out


# Model test for spectrogram
if __name__ == "__main__":
    patch_size = 16
    freq_bins = 128  # number of frequency bins in the spectrogram
    time_frames = 640  # number of time frames in the spectrogram
    num_patches = (freq_bins // patch_size) * (time_frames // patch_size)  # number of patches
    hidden_dim = 768  # hidden dimension
    token_mixing_dim = 384  # token mixing dimension
    channel_mixing_dim = 768  # channel mixing dimension
    num_of_mlp_blocks = 8
    num_classes = 10

    model = MLPMixer(patch_size, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim, num_of_mlp_blocks, freq_bins, time_frames, num_classes)

    # Test with a random batch of spectrograms
    spectrograms = torch.rand(32, 1, freq_bins, time_frames)  # (batch_size, 1 channel, freq_bins, time_frames)
    outputs = model(spectrograms)
    print("Output shape:", outputs.shape)  # Should be (batch_size, num_classes)
