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
        target = item['target']  # 이미 정수 형태의 클래스 인덱스

        # 오디오 파일 로드
        waveform, sr = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, target  # 정답을 정수로 반환

# 데이터셋 및 데이터 로더 설정
transform = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

train_dataset = AudioDataset('/home/kimdoyoung/AudioMAE/my_dataset/ESC50/meta/esc50_train.json', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델, 손실 함수, 최적화기 설정
model = MLPMixer(patch_size=16, 
                 num_patches=64, 
                 hidden_dim=768, 
                 token_mixing_dim=384, 
                 channel_mixing_dim=768, 
                 num_of_mlp_blocks=8, 
                 freq_bins=128, 
                 time_frames=640, 
                 num_classes=50).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 훈련 루프
model.train()
for epoch in range(100):  # 10 에폭으로 설정
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

print("Training complete!")
