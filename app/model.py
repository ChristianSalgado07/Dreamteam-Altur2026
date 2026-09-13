"""
Small CNN for the live AI-voice detector. Deliberately simple (hackathon
scope: prioritize a working prototype over an elaborate architecture) --
3 conv blocks + global average pool + 1 linear layer, fast enough for CPU
inference on a single 3-second log-Mel window.

Input:  (batch, 1, N_MELS, N_FRAMES) log-Mel spectrogram, see app/audio_utils.py
Output: (batch,) raw logit -- apply sigmoid for an AI-likelihood in [0, 1].
Convention (matches modeling/baseline.py): label 1 = synthetic/AI, 0 = human.
"""
import torch
import torch.nn as nn

# Pin to a single thread: multi-threaded CPU reduction order in
# conv/pooling ops can introduce tiny (usually ~1e-6) floating-point
# differences between runs. For a hackathon demo where "the same WAV
# must produce the same scores every time" matters more than shaving
# sub-millisecond latency off an already ~0.4ms/window budget, this
# trades essentially nothing for guaranteed determinism.
torch.set_num_threads(1)


class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)


def load_model(weights_path) -> VoiceCNN:
    model = VoiceCNN()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
