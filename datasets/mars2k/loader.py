import os
import tensorflow as tf
import glob
from tensorflow.python.data.experimental import AUTOTUNE



def image_dataset_from_directory(images_path):
    
    if not os.path.exists(images_path):
        print("Couldn't find directory: ", images_path)

    filenames = sorted(glob.glob(images_path + "/*.png"))
    print(len(filenames))

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)

    cache_directory = os.path.join(images_path, "cache", )

    os.makedirs(cache_directory, exist_ok=True)

    cache_file = cache_directory + "/cache"

    dataset = dataset.cache(cache_file)

    if not os.path.exists(cache_file + ".index"):
        populate_cache(dataset, cache_file)

    return dataset


def create_training_dataset(dataset_parameters, train_mappings, batch_size):
    lr_dataset = image_dataset_from_directory(dataset_parameters.train_directory)
    hr_dataset = image_dataset_from_directory('/content/drive/MyDrive/mars2k_dataset/HR/train')

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    for mapping in train_mappings:
        dataset = dataset.map(mapping, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_validation_dataset(dataset_parameters):
    lr_dataset = image_dataset_from_directory(dataset_parameters.valid_directory)
    hr_dataset = image_dataset_from_directory('/content/drive/MyDrive/mars2k_dataset/HR/valid')

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_training_and_validation_datasets(dataset_parameters, train_mappings, train_batch_size=16):
    training_dataset = create_training_dataset(dataset_parameters, train_mappings, train_batch_size)
    validation_dataset = create_validation_dataset(dataset_parameters)

    return training_dataset, validation_dataset


def populate_cache(dataset, cache_file):
    print(f'Begin caching in {cache_file}.')
    for _ in dataset: pass
    print(f'Completed caching in {cache_file}.')

