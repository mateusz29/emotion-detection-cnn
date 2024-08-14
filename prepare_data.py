import pandas as pd
import numpy as np
from PIL import Image
import os

def save_image_from_pixels(pixels: str, path: str) -> None:
    # Convert the pixels into a numpy array, reshape it and save it as an image
    img = np.array(pixels.split(' ')).astype(int).reshape(48, 48)
    img= Image.fromarray(img).convert('L')
    img.save(path)

def save_train_image_from_pixels(pixels: str, path: str) -> None:
    # Save the original image
    save_image_from_pixels(pixels, path)

    # Flip and rotate the image
    img = Image.open(path)
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip.save(f'{path[:-4]}_flip.png')
    img_rot = img.rotate(15, resample=Image.BICUBIC)
    img_rot.save(f'{path[:-4]}_15rot.png')
    img_rot = img.rotate(-15, resample=Image.BICUBIC)
    img_rot.save(f'{path[:-4]}_-15rot.png')

def main() -> None:
    # Load the dataset
    df = pd.read_csv('data/fer2013.csv')

    # Create the directories to save the images
    train_dir = 'data/train_images'
    test_dir = 'data/test_images'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Emotion labels
    class_names= ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Create subdirectories for each class
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Iterate over the rows of the dataset and save the images
    for i, row in df.iterrows():
        emotion = class_names[row['emotion']]
        pixels = row['pixels']
        usage = row['Usage']

        if usage == 'PublicTest' or usage == 'PrivateTest':
            image_path = os.path.join(test_dir, emotion, f'test_{i}.png')
            save_image_from_pixels(pixels, image_path)
        else:
            image_path = os.path.join(train_dir, emotion, f'train_{i}.png')
            save_train_image_from_pixels(pixels, image_path)

if __name__ == '__main__':
    main()