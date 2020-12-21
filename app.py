from flask import Flask, render_template,request
import pickle#Initialize the flask App

app = Flask(__name__)


from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # code to fetch uploaded images
    # then the clustering pipeline

    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)