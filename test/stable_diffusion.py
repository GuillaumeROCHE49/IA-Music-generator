import keras_cv
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512) # Le model de generation
diffusion_model = model.diffusion_model  # Le modèle de stable diffusion

images = model.text_to_image("A spectrogram of a sad hard rock music",
                             batch_size=1, num_steps=5) # type: list

print(type(images[0]))

def plot_images(images: list):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

plot_images(images)
plt.show()
