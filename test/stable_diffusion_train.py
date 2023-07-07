import keras_cv
import tensorflow as tf
import numpy as np
from classifier import Classifier

classifier = Classifier()
classifier.classify('music')
datas = classifier.get_datas()
training_set = []
for data in datas:
    name = data['name']
    spectrogram = np.array(data['spectrogram'])
    prompt = f"A {data['sub_class'][0]}-style {data['main_class']} spectrogram with {data['sub_class'][1]} and {data['sub_class'][2]} elements"
    training_set.append({
        'prompt': prompt,
        'spectrogram': spectrogram
    })

print("Loading model...")
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512) # Le model de generation
diffusion_model = model.diffusion_model  # Le modèle de stable diffusion

# Train the model on the training set using the diffusion model
diffusion_model.compile(optimizer='adam', loss='mse')
x_train = []
y_train = []

for set in training_set:
    prompt = set['prompt']
    spectrogram = set['spectrogram']
    x_train.append(prompt)
    y_train.append(spectrogram)

dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
diffusion_model.fit(dataset, epochs=10)


