
import keras_cv
import matplotlib.pyplot as plt
from classifier import Classifier

classifier = Classifier()
classifier.classify('music')
datas = classifier.get_datas()
training_set = []
for data in datas:
    name = data['name']
    spectrogram = data['spectrogram']
    prompt = f"A {data['sub_class'][0]}-style {data['main_class']} spectrogram with {data['sub_class'][1]} and {data['sub_class'][2]} elements"
    training_set.append({
        'name': name,
        'prompt': prompt,
        'spectrogram': spectrogram
    })

print(training_set)

'''
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512) # Le model de generation
diffusion_model = model.diffusion_model  # Le modèle de stable diffusion

print(diffusion_model.get_layer("spatial_transformer_7"))
diffusion_model.compile()
diffusion_model.fit()
'''
