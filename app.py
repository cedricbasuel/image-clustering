from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import os
import pandas as pd
# from functools import cached_property 
import tensorflow as tf

from image_clustering import load_images, get_embedding, cluster_images, get_embedding_only

# @cached_property
def load_model(modelname, image_shape):
    if modelname == 'mobilenet':
        model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=True, ### wait True nga dapat db?? nalito ako
            input_shape=image_shape,
            layers=tf.keras.layers
        )
    elif modelname=='resnet50':
        model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=True, 
            input_shape=image_shape
        )
    else:
        raise ValueError('Model not yet available.')
    return model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

model_resnet = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=True, 
            input_shape=(224,224,3)
        )

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'


@app.route('/', methods=['POST', 'GET'])
def index():

    if request.method=='POST':
        for img in request.files.getlist('file'):
            img.save(os.path.join(app.config['UPLOAD_PATH'], img.filename))
        return redirect(url_for('index'))
    
    images = os.listdir(app.config['UPLOAD_PATH'])

    return render_template('index.html', images=images)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'],filename)



@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method=='POST':

        num_cluster = request.form.get('num_cluster', False)
        modelname = request.form.get('modelname')
        print("MODELNAME",modelname)

        num_cluster = int(num_cluster)

        # model = load_model(modelname, image_shape=(224, 224, 3))

        if modelname == 'resnet50':
            model = model_resnet
        # elif modelname == 'mobilenet':
        #     model = model_mobilenet
        else:
            raise ValueError('Model not yet available.')
        
        _labels = {'animalz': 0}

        image_list, label_list = load_images(dir=app.config['UPLOAD_PATH'],
            target_size=(224,224),
            labelmap=_labels
            )

        image_list, emb_list = get_embedding_only(image_list=image_list,
        model_name=model,
        image_shape=(224,224,3))

        # image_list, emb_list = get_embedding(image_list=image_list,
        # model_name=model,
        # image_shape=(224,224,3))

        clustered_images = cluster_images(image_list=image_list, 
        emb_list=emb_list, 
        num_clusters=num_cluster
        )
        
        # clean this up laterrrr
        cluster_df = pd.DataFrame({'Image':label_list, 'Cluster':clustered_images})

        cluster_dict = {}
        cluster_dict = {key: None for key in list(set(cluster_df['Cluster']))}

        for key in cluster_dict.keys():
            temp_df = cluster_df[cluster_df['Cluster']==key]
            cluster_dict[key] = list(temp_df['Image'])


        images = os.listdir(app.config['UPLOAD_PATH'])
        
    return render_template('index.html', images=images, cluster_dict=cluster_dict)


if __name__=="__main__":
    app.run(debug=True)




# sudo rm -rf ~/.nv/