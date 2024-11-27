import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import warnings
from torchaudio.transforms import MelSpectrogram

# Importing the MLPMixer model from your defined path
from MLPMixer_audio import MLPMixer

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 데이터셋 클래스 정의
class AudioDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = item['path']
        target = torch.zeros(50)
        target[item['target']]=1  # 이미 정수 형태의 클래스 인덱스
        # print(f"target : {target.shape}")
        # 오디오 파일 로드
        y, sr = torchaudio.load(audio_path)
        if y.ndim > 1:  # Check for multiple channels
            y = y[0, :].squeeze(0)
        # print(f"raw audio : {y.shape}")

        num_samples = sr * 5
        if len(y) < num_samples:
            padding = num_samples -len(y)
            y = torch.nn.functional.pad(y, (0, padding), mode='constant')
        else:
            y = y[:num_samples]
        # print(f"pdded audio : {y.shape}")

        mel_spec = self.transform(y)
        # print(f"mel_spec : {mel_spec.shape}")

        # Convert to log scale
        log_mel_spec = T.AmplitudeToDB()(mel_spec)
        # print(f"log_mel_spec : {log_mel_spec.shape}")
        spec_tensor = log_mel_spec.unsqueeze(0).float()  # Add batch dimension for consistency
        # print(f"spec_tensor : {spec_tensor.shape}")

        h_pad = (16 - spec_tensor.shape[1] % 16) % 16
        w_pad = (16 - spec_tensor.shape[2] % 16) % 16
        spec_tensor = F.pad(spec_tensor, (0, w_pad, 0, h_pad), mode='constant', value=0)
        
        return spec_tensor, target
    
# 데이터셋 및 데이터 로더 설정
transform = MelSpectrogram(
    sample_rate=44100,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

batch_size = 128
train_dataset = AudioDataset('/home/kimdoyoung/AudioMAE/my_dataset/ESC50/meta/esc50_train.json', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

# 모델, 손실 함수, 최적화기 설정
patch_size          = 16
freq_bins           = 128
time_frames         = 432
num_patches         = (freq_bins // patch_size) * (time_frames // patch_size)
hidden_dim          = 768
token_mixing_dim    = 384
channel_mixing_dim  = 768
num_of_mlp_blocks   = 8
num_classes         = 50

gpu = "cuda:0"
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

model = MLPMixer(
    patch_size=patch_size, 
    num_patches=num_patches, 
    hidden_dim=hidden_dim,
    token_mixing_dim=token_mixing_dim, 
    channel_mixing_dim=channel_mixing_dim, 
    num_of_mlp_blocks=num_of_mlp_blocks, 
    freq_bins=freq_bins, 
    time_frames=time_frames, 
    num_classes=num_classes,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 훈련 루프
model.train()
for epoch in range(100):  # 10 에폭으로 설정
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        # print(f"input, label = {inputs.shape}, {labels.shape}")
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f"output = {outputs.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

print("Training complete!")
