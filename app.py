from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
import os
import pandas as pd


from image_clustering import load_images, get_embedding, cluster_images

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/', methods=['POST', 'GET'])
def index():
    
    # uri=None
    if request.method=='POST':

        # if True:
        #     img = request.files['file']
        for img in request.files.getlist('file'):
            img.save(os.path.join(app.config['UPLOAD_PATH'], img.filename))
            # img1 = np.frombuffer(img.read(), np.uint8)
            # img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
            # img1 = Image.fromarray(img1.astype("uint8"))
            # rawBytes = io.BytesIO()
            # img1.save(rawBytes, "JPEG")
            # rawBytes.seek(0)
            # img1_base64 = b64encode(rawBytes.getvalue()).decode('utf-8') 

            # encoded = b64encode(img.stream)
            # mime = "image/jpeg"
            # uri = "data:%s;base64,%s" % (mime, img1_base64)
            
            # images.append(uri)

        return redirect(url_for('index'))
    
    images = os.listdir(app.config['UPLOAD_PATH'])





    return render_template('index.html', images=images)
    # return render_template('index.html',uri=uri)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'],filename)



# @app.route('/predict', methods=['POST', 'GET'])
# def predict(dir):

#     _labels = {'animalz': 0}

#     image_list, label_list = load_images(dir=dir,
#         target_size=(224,224),
#         labelmap=_labels
#         )

#     image_list, emb_list = get_embedding(image_list=image_list,
#     model_name='mobilenet',
#     image_shape=(224,224,3))

#     # print(emb_list.shape)

#     clustered_images = cluster_images(image_list=image_list, 
#     emb_list=emb_list, 
#     num_clusters=5
#     )
    
#     # clean this up laterrrr
#     cluster_df = pd.DataFrame({'Image':label_list, 'Cluster':clustered_images})

#     cluster_dict = {}
#     cluster_dict = {None for key in list(set(cluster_df['cluster']))}
#     for key in cluster_dict.keys():
#         temp_df = cluster_df[cluster_df['Cluster']==key]
#         cluster_dict[key] = list(temp_df['Image'])

#     return render_template('index.html',cluster_dict=cluster_dict)


if __name__=="__main__":
    app.run(debug=True)