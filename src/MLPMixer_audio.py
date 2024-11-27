import torch
import torch.nn as nn
import torchaudio.transforms as T
import pdb
import time 

class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, spectrograms):
        st = time.perf_counter()
        batch_size = spectrograms.size(0)
        # Extract patches using unfold for spectrograms

        patches = spectrograms.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * spectrograms.size(1))
        # print(f"Patches forward time: {(time.perf_counter() - st):.3f}")
        return patches

class MLPBlock(nn.Module):
    '''
    num_patches, hidden_dim = S, C
    '''
    def __init__(self, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim):
        super(MLPBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        
        # Token Mixing
        self.W1 = nn.Linear(num_patches, token_mixing_dim)
        self.W2 = nn.Linear(token_mixing_dim, num_patches)
        
        # Channel Mixing
        self.W3 = nn.Linear(hidden_dim, channel_mixing_dim)
        self.W4 = nn.Linear(channel_mixing_dim, hidden_dim)
        
        # Cross-attention 추가해봄
        # 주파수-시간 축 상호작용 학습 강화
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        
    def forward(self, X):                                           # X shape: (batch_size, num_patches, hidden_dim)
        st = time.perf_counter()

        # Token Mixing
        X_T = self.layer_norm1(X).transpose(1, 2)                   # (batch_size, hidden_dim, num_patches)

        print(f"X : {X.shape}")
        print(f"X_T : {X_T.shape}")
        print(f"self.W1(X_T) : {self.W1(X_T).shape}")
        print(f"self.gelu(self.W1(X_T)) : {self.gelu(self.W1(X_T)).shape}")
        print(f"self.W2(self.gelu(self.W1(X_T))) : {self.W2(self.gelu(self.W1(X_T))).shape}")

        U = self.W2(self.gelu(self.W1(X_T))).transpose(1, 2) + X    # Skip connection & (batch_size, num_patches, hidden_dim)

        # # Cross-Attention 추가
        # attn_out, _ = self.cross_attention(U, U, U)  # Self-attention 형태로 학습
        # U = attn_out + U
        
        # Channel Mixing
        Y = self.W4(self.gelu(self.W3(self.layer_norm2(U)))) + U    # Skip connection & (batch_size, num_patches, hidden_dim)
        # print(f"MLP Block forward time: {(time.perf_counter() - st):.3f}")

        return Y                                                    # (batch_size, num_patches, hidden_dim)


class MLPMixer(nn.Module):
    def __init__(self, patch_size, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim, num_of_mlp_blocks, freq_bins, time_frames, num_classes):
        super(MLPMixer, self).__init__()
        self.projection = nn.Linear(patch_size * patch_size, hidden_dim)
        self.mlp_blocks = nn.ModuleList(
            [MLPBlock(num_patches, 
                      hidden_dim, 
                      token_mixing_dim, 
                      channel_mixing_dim) for _ in range(num_of_mlp_blocks)]
        )
        
        # Spectrogram only Augmentation
        self.data_augmentation = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=30),
            nn.Dropout(0.1),
        )

        self.classification_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),               # Layer normalization
            nn.Linear(hidden_dim, num_classes),     # Fully connected layer to map to number of classes
            nn.Dropout(0.2),                        # Dropout
            # nn.Softmax(dim=1)                     # Softmax to output class probabilities # Softmax는 모델 밖에서 하는게 관례적
        )
        self.patches = Patches(patch_size)

    def forward(self, spectrograms):
        st = time.perf_counter()
        batch_size = spectrograms.size(0)
        
        augmented_spectrograms = self.data_augmentation(spectrograms) # Data augmentation
        print(f"augmented_spec = {augmented_spectrograms.shape}")
        X = self.patches(augmented_spectrograms) # Extract patches from spectrograms
        print(f"patches = {X.shape}")
        X = self.projection(X) # Per-patch Fully-connected
        print(f"projected X = {X.shape}")

        # MLP Blocks
        for block in self.mlp_blocks:
            X = block(X)
        print(f"Blocked X = {X.shape}")

        # Global average pooling across patches
        X = X.mean(dim=1)  # (batch_size, hidden_dim)
        ''' # 대안 1.
        X_avg = X.mean(dim=1)  # (batch_size, hidden_dim)
        X_max = X.max(dim=1).values  # Global Max Pooling
        X = torch.cat([X_avg, X_max], dim=1)  # Concatenate pooled features'''
        '''# 대안 2.
        X_avg = X.mean(dim=1, keepdim=True)  # GAP with the same dimension
        X_max = X.max(dim=1, keepdim=True).values  # Max Pooling with the same dimension
        X = X + X_avg + X_max  # Combine while keeping dimensions consistent'''
        print(f"GAP ed X = {X.shape}")

        # Classification layer
        out = self.classification_layer(X)
        print(f"out = {out.shape}")

        # print(f"MLP Mixer forward time: {(time.perf_counter() - st):.3f}")
        return out


# Model test for spectrogram
if __name__ == "__main__":
    patch_size = 16
    freq_bins = 128         # number of frequency bins in the spectrogram
    time_frames = 640       # number of time frames in the spectrogram
    num_patches = (freq_bins // patch_size) * (time_frames // patch_size)  # number of patches
    hidden_dim = 768  # hidden dimension
    token_mixing_dim = 384  # token mixing dimension
    channel_mixing_dim = 768  # channel mixing dimension
    num_of_mlp_blocks = 8
    num_classes = 527

    model = MLPMixer(patch_size, num_patches, hidden_dim, token_mixing_dim, channel_mixing_dim, num_of_mlp_blocks, freq_bins, time_frames, num_classes)

    # Test with a random batch of spectrograms
    spectrograms = torch.rand(32, 1, freq_bins, time_frames)  # (batch_size, 1 channel, freq_bins, time_frames)
    outputs = model(spectrograms)
    print("Output shape:", outputs.shape)  # Should be (batch_size, num_classes)()