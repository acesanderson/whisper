"""
We will use Bark.
pip install --upgrade pip
pip install --upgrade transformers scipy
"""

# from transformers import pipeline
# import scipy
#
# synthesizer = pipeline("text-to-speech", "suno/bark")
# speech = synthesizer("We are living la vida loca", forward_params={"do_sample": True})
# scipy.io.wavfile.write(
#     "bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"]
# )

#
# from transformers import AutoProcessor, AutoModel
#
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModel.from_pretrained("suno/bark")
#
# inputs = processor(
#     text=["We are living la vida loca"],
#     return_tensors="pt",
# )
# speech_values = model.generate(**inputs, do_sample=True)
#
# import scipy
#
# sampling_rate = model.config.sample_rate
# scipy.io.wavfile.write(
#     "bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze()
# )
#


from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

inputs = processor(
    text=[
        "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."
    ],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

import scipy

sampling_rate = model.config.sample_rate
scipy.io.wavfile.write(
    "bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze()
)
