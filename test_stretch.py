import numpy as np
import librosa
audio = np.random.randn(16000).astype(np.float32)
stretched = librosa.effects.time_stretch(audio, rate=1.5)
print(len(stretched) / 16000, len(audio) / 16000)
