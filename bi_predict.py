import os
import torch
from model_def import MiniVGGClassifier, AudioUtil

DURATION = 4000
SR = 44100
CHANNEL = 2
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", 
    "siren", "street_music"
]

device = torch.device("cpu")
model = MiniVGGClassifier()
model.load_state_dict(torch.load("model2.pt", map_location=device))
model.eval()

def get_prediction(audio_filepath):
    # 1. Предобработка аудиопотока
    aud = AudioUtil.open(audio_filepath)
    reaud = AudioUtil.resample(aud, newsr=SR)
    rechan = AudioUtil.rechannel(reaud, new_channel=CHANNEL)
    dur_aud = AudioUtil.pad_trunc(rechan, max_ms=DURATION)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

    sgram = sgram.unsqueeze(0).to(device)
    sgram_m, sgram_s = sgram.mean(), sgram.std()
    sgram = (sgram - sgram_m) / sgram_s

    with torch.no_grad():
        outputs = model(sgram)
        prediction = torch.argmax(outputs, dim=1).item()

    return CLASSES[prediction]