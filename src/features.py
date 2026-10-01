import math
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

def extract_spectrogram(waveform, sr, max_frames=600):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    n_fft = 1724
    win_length = 1724
    hop_length = int(0.0081 * sr)
    window = torch.blackman_window(win_length)
    
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=lambda _: window,
        power=2.0
    )
    
    spec = spectrogram_transform(waveform)
    log_spec = torch.log(spec + 1e-9)
    time_frames = log_spec.shape[-1]
    
    if time_frames < max_frames:
        pad_amount = max_frames - time_frames
        log_spec = F.pad(log_spec, (0, pad_amount), value=math.log(1e-9))
    else:
        log_spec = log_spec[:, :, :max_frames]
        
    return log_spec