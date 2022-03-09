import os

from skimage.io import imread, imsave
from skimage import img_as_ubyte, transform
import numpy as np

IMAGE_FILE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

def list_folders(root):
    for dir in os.listdir(root):
        if os.path.isdir(os.path.join(root, dir)):
            yield dir

def get_samples(root_dir, images_folder="images", masks_folder="masks", unlabeled=False):
    images = []
    for dirpath, _, filenames in os.walk(os.path.join(root_dir, images_folder)):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in IMAGE_FILE_EXTENSIONS):
                images.append(os.path.relpath(os.path.join(dirpath, filename), start=os.path.join(root_dir, images_folder)))
    samples = []
    for image_path in images:
        if unlabeled:
            samples.append({
                "image_path": os.path.join(root_dir, images_folder, image_path)
            })
        elif os.path.exists(os.path.join(root_dir, masks_folder, image_path)):
            samples.append({
                "image_path": os.path.join(root_dir, images_folder, image_path),
                "mask_path": os.path.join(root_dir, masks_folder, image_path)
            })
    
    return samples

def load_image(path):
    return imread(path)

def save_image(path, image):
    imsave(path, image)

def resize_image(image, output_size):
    return img_as_ubyte(transform.resize(image, output_size))

def resize_mask(mask, output_size):
    return img_as_ubyte(transform.resize(mask, output_size, order=0, anti_aliasing=False))

def load_index_file(path, unlabeled=False):
    return np.loadtxt(path, dtype=str, delimiter=' ', usecols=0 if unlabeled else (0, 1))

def get_color_map(data_config):
    color_map = np.ndarray((256 * 256 * 256), dtype="int32")
    color_map[:] = -1
    for label in data_config["labels"]:
        color_map[np.dot(label["color"], [1, 256, 256**2])] = label["id"]
    return color_map

def get_id_to_label(data_config):
    return np.vectorize({
        _class["id"]: _class["label"] for _class in data_config["labels"]
    }.get)