import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

filename = '0ad98688a86634871324bdd20080a74a'
file = filename + '.wav'
samples, sample_rate = librosa.load('C:\\dev\\tfg\\src\\poc\\toses\\DETECTABLES\\' + file)

fig = plt.figure(figsize=[1,1]) #4,4 original value
plt.axis('off')

S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)

canvas = FigureCanvas(fig)
p = librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
fig.savefig(filename + '.png', bbox_inches='tight',pad_inches = 0)