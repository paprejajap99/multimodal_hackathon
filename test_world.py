import numpy as np
import pyworld

audio = np.random.randn(16000).astype(np.float64)
f0, t = pyworld.dio(audio, 16000)
f0 = pyworld.stonemask(audio, f0, t, 16000)
sp = pyworld.cheaptrick(audio, f0, t, 16000)
ap = pyworld.d4c(audio, f0, t, 16000)
synth = pyworld.synthesize(f0, sp, ap, 16000).astype(np.float32)

print(synth.shape, audio.shape)
