import os
import torch
from russian_stt_text_normalization.normalizer import Normalizer

device = torch.device("cpu")
torch.set_num_threads(4)
tts_ru_model = "tts_ru.pt"
tts_en_model = "tts_en.pt"


def download_model(model):
    if not os.path.isfile(model):
        torch.hub.download_url_to_file(
            "https://models.silero.ai/models/tts/ru/v3_1_ru.pt", model
        )


model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = "В недрах тундры выдры в гетрах тырят в вёдра ядра кедров."
sample_rate = 48000
speaker = "baya"

norm = Normalizer()
result = norm.norm_text(example_text)


audio_paths = model.save_wav(text=result, speaker=speaker, sample_rate=sample_rate)

print(audio_paths)
