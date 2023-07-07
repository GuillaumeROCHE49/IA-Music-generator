from custom_riffusion import riffusion
from custom_riffusion.spectrogram.spectrogram_params import SpectrogramParams
from util.classifier import Classifier

import torch

prompt = "a dark blues with drum"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Generate music using Riffusion (https://github.com/riffusion/riffusion)
img = riffusion.run_txt2img(
    prompt=prompt,
    negative_prompt="classic",
    num_inference_steps=10,
    guidance=7.0,
    seed=42,
    width=512,
    height=512,
    checkpoint="riffusion/riffusion-model-v1",
    device=device,
    scheduler="DPMSolverMultistepScheduler"
)
audio = riffusion.audio_segment_from_spectrogram_image(
    image=img,
    params=SpectrogramParams(
        stereo=False,
        sample_rate=44100,
        step_size_ms=10,
        window_duration_ms=100,
        padded_duration_ms=400,
        num_frequencies=512,
        min_frequency=0,
        max_frequency=10000,
        mel_scale_norm=None,
        mel_scale_type="htk",
        max_mel_iters=200,
        num_griffin_lim_iters=32,
        power_for_image=0.25,
    ),
    device=device
)
# Save music to file
audio.export("output.wav", format="wav")
# Classify music using YAMNet
classifier = Classifier()
classifier.classify_single("output.wav")
print(classifier)
