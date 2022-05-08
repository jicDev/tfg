import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

filename = 'tos-test-0011'
file = filename + '.wav'
samples, sample_rate = librosa.load(file)

fig = plt.figure(figsize=[4,4])
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

canvas = FigureCanvas(fig)
p = librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
fig.savefig(filename + '.png')