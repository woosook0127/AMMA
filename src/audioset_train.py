import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from MLPMixer_audio import MLPMixer  # MLP-Mixer for Audio model
from sklearn.metrics import average_precision_score, f1_score

from tqdm import tqdm
import warnings
import pdb
import time 
# librosa의 FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# PySoundFile 관련 UserWarning 억제
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

STFT_init = {
    'nWin': 1024,
    'nfft': 1024,
    'nShift': 256,
    'sr':16000,
    'n_mels': 128
}

def calculate_metrics(outputs, labels):
    outputs = torch.sigmoid(outputs).cpu().numpy()  # Sigmoid 적용 후 확률 변환
    labels = labels.cpu().numpy()
    
    mAP = average_precision_score(labels, outputs, average="macro")  # mAP 계산
    micro_f1 = f1_score(labels, outputs > 0.5, average="micro")
    macro_f1 = f1_score(labels, outputs > 0.5, average="macro")
    
    return {"mAP": mAP, "Micro F1": micro_f1, "Macro F1": macro_f1}

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    st = time.perf_counter()

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    # print(f"############ Add weight decay time: {(time.perf_counter() - st):.3f}")

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def load_metadata(json_path):
    st = time.perf_counter()

    with open(json_path, 'r') as f:
        metadata = json.load(f)
    # print(f"############ Load Metadata time: {(time.perf_counter() - st):.3f}")

    return list(metadata.values())

def load_label_map(csv_path):
    st = time.perf_counter()
    df = pd.read_csv(csv_path)
    label_map = {row['mid']: row['index'] for _, row in df.iterrows()}
    # print(f"############ Load Lable Map time: {(time.perf_counter() - st):.3f}")

    return label_map

# 데이터셋 클래스
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, data_folder, label_map, transform=None):
        problematic_files = [
            "wavs_resampled/balanced_train/000002/id_cHfPCPrffSQ.wav",
            "wavs_resampled/unbalanced_train/000076/id_5ym3QWL2bRM.wav",
            "wavs_resampled/unbalanced_train/000130/id_CmgWyoNh_LU.wav",
            "wavs_resampled/unbalanced_train/000182/id_Klq-we49OVU.wav",
            "wavs_resampled/unbalanced_train/000321/id_iMMwB0h3v20.wav",
            "wavs_resampled/unbalanced_train/000390/id_wMjEr9oDnJk.wav",
        ]
        self.metadata = metadata
        self.metadata = [m for m in metadata if m['path'] not in problematic_files ]
        self.data_folder = data_folder
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.data_folder, item["path"])
        tags = item["tags"]
        
        # Convert tags to multi-hot vector
        st = time.perf_counter()
        label_tensor = torch.zeros(len(self.label_map))
        for tag in tags:
            if tag in self.label_map:
                label_tensor[self.label_map[tag]] = 1
        # print(f"##GET ITEM## Tag to Multi-hot vec: {(time.perf_counter() - st):.3f}, label_tensor: {label_tensor.shape}")

        # Load audio with librosa and convert to Tensor
        st = time.perf_counter()
        y, sr = torchaudio.load(audio_path)  # Load the audio
        # Convert to mono 
        if y.ndim > 1:  # Check for multiple channels
            y = y[0, :].squeeze(0)
        # print(f"##GET ITEM## load audio: {(time.perf_counter() - st):.3f}")        

        # Pad or truncate to 10 seconds (16000 * 10 samples)
        st = time.perf_counter()
        num_samples = STFT_init['sr'] * 10
        if len(y) < num_samples:
            padding = num_samples -len(y)
            y = torch.nn.functional.pad(y, (0, padding), mode='constant')
        else:
            y = y[:num_samples]
        # print(f"##GET ITEM## padding time: {(time.perf_counter() - st):.3f}")

        # Compute MelSpectrogram
        st = time.perf_counter()
        mel_spec = self.transform(y)
        # Convert to log scale
        log_mel_spec = T.AmplitudeToDB()(mel_spec)
        spec_tensor = log_mel_spec.unsqueeze(0).float()  # Add batch dimension for consistency
        # print(f"##GET ITEM## convert log mel scale: {(time.perf_counter() - st):.3f}, spec_tensor: {spec_tensor.shape}")

        return spec_tensor, label_tensor


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training Batch", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# 검증 함수
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation Batch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# 주요 학습 코드
from tensorboardX import SummaryWriter

'''For AudioSet'''
if __name__ == "__main__":
    dset_root = "/home/nas3/DB/Audioset/Audioset_16k_wav/"
    # Paths
    balanced_train_json =   dset_root + "audioset_balanced_train_metadata.json"
    unbalanced_train_json = dset_root + "audioset_unbalanced_train_metadata.json"
    eval_json = dset_root + "audioset_eval_metadata.json"
    csv_path =  dset_root + "class_labels_indices.csv"
    data_folder = dset_root # Root folder containing balanced_train, eval, unbalanced_t

    log_dir = "../output/logs/mlp_mixer_audio"
    writer = SummaryWriter(log_dir)

    # Device 설정
    gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    # Load label map
    label_map = load_label_map(csv_path)
    
    # Load metadata
    balanced_metadata = load_metadata(balanced_train_json)
    unbalanced_metadata = load_metadata(unbalanced_train_json)
    eval_metadata = load_metadata(eval_json)
    
    # Hyperparameters
    batch_size          = 128
    epochs              = 32
    learning_rate       = 1e-4

    patch_size          = 16
    freq_bins           = 128
    time_frames         = 640
    num_patches         = (freq_bins // patch_size) * (time_frames // patch_size)
    hidden_dim          = 768
    token_mixing_dim    = 384
    channel_mixing_dim  = 768
    num_of_mlp_blocks   = 8
    num_classes         = len(label_map)
    
    # 데이터 변환
    transform = T.MelSpectrogram(
        sample_rate =STFT_init['sr'],
        n_fft       =STFT_init['nfft'],
        hop_length  =STFT_init['nShift'],
        n_mels      =STFT_init['n_mels']
    )

    # Dataset and DataLoader
    balanced_dataset = AudioDataset(balanced_metadata, data_folder, label_map, transform)
    unbalanced_dataset = AudioDataset(unbalanced_metadata, data_folder, label_map, transform)
    eval_dataset = AudioDataset(eval_metadata, data_folder, label_map, transform)
    train_dataset = ConcatDataset([balanced_dataset, unbalanced_dataset])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    # Model Initialization
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
    
    # 손실 함수 및 옵티마이저 
    # criterion = nn.CrossEntropyLoss() # For multi-class
    criterion = nn.BCEWithLogitsLoss() # For multi-label
    ''' For biased class data '''
    # class_counts = torch.tensor([metadata.count(c) for c in label_map.keys()])
    # pos_weight = (1.0 / class_counts).to(device)  # 반비례 가중치
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight_decay = 0.05
    param_groups = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        anneal_strategy="cos",
        cycle_momentum=False
    )
        
    # Step-based training
    step = 0
    with tqdm(total=total_steps, desc="Training MLP Mixer for Audio") as pbar:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    continue
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    continue
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()
                step += 1

                # TensorBoard에 손실 기록
                writer.add_scalar("Loss/Train", loss.item(), step)

                # Progress bar 업데이트
                pbar.update(1)
                pbar.set_postfix({"Train Loss": loss.item()})

                # 모델 저장
                if step % 500 == 0:  # 500 step마다 저장
                    output_dir = "../output"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(model.state_dict(), f"{output_dir}/mlp_mixer_epoch_{step:08d}.pth")

            # Validation at the end of each epoch
            eval_loss = validate(model, eval_loader, criterion, device)
            writer.add_scalar("Loss/Eval", eval_loss, step)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / steps_per_epoch:.4f}, Eval Loss: {eval_loss:.4f}")

    torch.save(model.state_dict(), f"{output_dir}/mlp_mixer_epoch_{step:08d}.pth")
    print("Training Complete!")