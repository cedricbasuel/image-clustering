'''Script for  image clustering using CNNs.

Usage:
    python image_clustering.py <config>.yaml

Author:
    Cedric Basuel

'''

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
        List of raw images.
    
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
def get_embedding(image_list, model_name, image_shape, model_loaded=True):
    '''Use a pretrained CNN model to get features from an image.

    Params
    ------
    image_list : ndarray
        List of raw images.

    model_name : str
        Name of CNN model.

    image_shape : tuple
        Dimensions of image.

    model_loaded : bool
        True if  model is preloaded in env, otherwise load CNN model from scratch.

    Returns
    -------
    emb_list : ndarray
        Extracted image features.

    '''

    emb_list = []

    logging.info('Getting image embeddings...')
    logging.info(model_name)

    model = model_name

    if not model_loaded:
        if model_name=='mobilenet':
            model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True, ### wait True nga dapat db?? nalito ako
                input_shape=image_shape,
                layers=tf.keras.layers
            )
        
        elif model_name=='resnet50':
            model = tf.keras.applications.ResNet50(
                weights='imagenet', 
                include_top=True, 
                input_shape=image_shape
            )

        else: 
            raise ValueError('Model not yet available.')

        logging.info('Successfully loaded model.')


    for img in image_list:
        img = np.expand_dims(img, axis=0)
        temp_emb = model.predict(img)
        emb_list.append(np.squeeze(temp_emb))
    
    logging.info('Successfully extracted embeddings.')
    
    return image_list, np.array(emb_list)


# remove!!!
@timer
def get_embedding_only(image_list, model_name, image_shape):
    emb_list = []

    logging.info('Getting image embeddings...')
    model = model_name

    for img in image_list:
        img = np.expand_dims(img, axis=0)
        temp_emb = model.predict(img)
        emb_list.append(np.squeeze(temp_emb))
    logging.info('successfully extracted embeddings!!')

    return image_list, np.array(emb_list)


@timer
def cluster_images(image_list, emb_list, num_clusters):
    '''Use kmeans clustering to from explore clusters of images
    given the n-dimensional embeddings from get_embedding().

    Params
    ------
    emb_list : ndarray
        Embeddings extracted from each image.

    num_clusters: int
        Number of clusters to fit.

    Returns
    -------
    clustered_images : list
        List of integers indicating cluster membership for each image.

    '''
    kmodel = KMeans(n_clusters=num_clusters, n_jobs=-1)

    logging.info(num_clusters)
    logging.info(emb_list.shape)

    kmodel.fit(emb_list)
    logging.info('KMeans clustering done.')

    clustered_images = kmodel.predict(emb_list)

    return clustered_images




if __name__=='__main__':


    CONFIG_FILE = sys.argv[1]

    with open(CONFIG_FILE) as cfg:
        config = yaml.safe_load(cfg)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            pass

    _labels = config['training']['labels']
    image_shape = config['training']['image_shape']

    image_list, label_list = load_images(dir=config['training']['dir'],
        target_size=(image_shape, image_shape),
        labelmap=config['training']['labels']
        )

    image_list, emb_list = get_embedding(image_list=image_list,
    model_name=config['training']['model_name'],
    image_shape=(image_shape, image_shape, 3),
    model_loaded=False
    )

    logging.info(emb_list.shape)

    clustered_images = cluster_images(image_list=image_list, 
    emb_list=emb_list, 
    num_clusters=config['training']['num_clusters']
    )
    
    logging.info('Done!')
    df = pd.DataFrame({'Image' : label_list, 'Cluster' : clustered_images})

    print(df)