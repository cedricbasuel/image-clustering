import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import yaml
import time
import logging
from functools import wraps
from sklearn.cluster import KMeans
import pandas as  pd


logging.basicConfig(
    # filename='train_image.log',
    format='[IMAGE_CLUSTERING] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )

def timer(func):
    @wraps(func)
    def wrapper_time(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        run_time = time.time() - start_time
        logging.info('{} took {} seconds to finish.'.format(func.__name__, run_time))
        return value
    return wrapper_time

@timer
def load_images(dir, target_size, labelmap):
    '''Load train/test images from a directory.

    Params
    ------
    dir : str
        Where images will be loaded from.

    Returns
    -------
    image_list : ndarray
    
    label_list : ndarray
        Label list for each image, mapped to integer labels.

    '''
    
    image_list = []
    label_list = []
    logging.info('Loading images from {}...'.format(dir))
    for root, dirs, files in os.walk(dir):
        for file in files:
            temp_dir = str.split(root, '/')
            
            try:
                temp_image = cv2.imread(os.path.join(root, file))
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                temp_image = cv2.resize(temp_image, dsize=target_size)
                image_list.append(temp_image)

                # folder name is class name
                # changed: im just getting the file name here since unsupervised
                label_list.append(file)
            
            except cv2.error:
                pass

    image_list = np.array(image_list)
    # label_list = np.array(
        # [labelmap[label] for label in label_list], 
        # dtype='int32'
    # )

    print('Image list:', image_list.shape)
    # print('Label list:', label_list.shape)

    return image_list, label_list

@timer
def get_embedding(image_list, model_name, image_shape):
    emb_list = []

    logging.info('Getting image embeddings...')
    # logging.info(model_name)
    if model_name=='mobilenet':
        model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True, ### wait True nga dapat db?? nalito ako
            input_shape=image_shape,
            layers=tf.keras.layers
        )

    for img in image_list:
        img = np.expand_dims(img, axis=0)
        temp_emb = model.predict(img)
        emb_list.append(np.squeeze(temp_emb))

    return image_list, np.array(emb_list)


@timer
def cluster_images(image_list, emb_list, num_clusters):
    kmodel = KMeans(n_clusters=num_clusters, n_jobs=-1)
    
    kmodel.fit(emb_list)
    logging.info('KMeans clustering done.')

    clustered_images = kmodel.predict(emb_list)

    return clustered_images




if __name__=='__main__':

    _labels = {'animalz': 0}

    image_list, label_list = load_images(dir='/home/cedric/Downloads/photos',
        target_size=(224,224),
        labelmap=_labels
        )

    image_list, emb_list = get_embedding(image_list=image_list,
    model_name='mobilenet',
    image_shape=(224,224,3))

    # print(emb_list.shape)

    clustered_images = cluster_images(image_list=image_list, 
    emb_list=emb_list, 
    num_clusters=5
    )
    
    logging.info('Done!')
    df = pd.DataFrame({'Image':label_list, 'Cluster':clustered_images})

    print(df)