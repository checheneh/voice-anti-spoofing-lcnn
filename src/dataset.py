from src.features import extract_spectrogram
import torch
import os
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

class ASVspoofDataset(Dataset):
    def __init__(self, protocol_file, audio_dir, max_frames=600, is_eval=False):
        self.audio_dir = audio_dir
        self.max_frames = max_frames
        self.is_eval = is_eval  
        self.data = []
        
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            filename = parts[1] + '.flac'
            
            if not self.is_eval:
                label_text = parts[4]
                label = 0 if label_text == 'bonafide' else 1
            else:
                label = -1
                
            self.data.append((filename, label))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename, label = self.data[idx]
        audio_path = os.path.join(self.audio_dir, filename)
        waveform, sr = torchaudio.load(audio_path)
        features = extract_spectrogram(waveform, sr, self.max_frames)
        return features, torch.tensor(label, dtype=torch.long), filename
