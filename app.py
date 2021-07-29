from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# load the model from disk
filename = 'C:\\Users\\hp\\Documents\\Portfolio Projects\\Sentiment Analysis\\tsa.pkl'
#clf = pickle.load(open(filename, 'rb'))
filename1 = 'C:\\Users\\hp\\Documents\\Portfolio Projects\\Sentiment Analysis\\transform.pkl'
#cv = pickle.load(open(filename1,'rb'))
with open(filename,'rb') as f:
	a = pickle.Unpickler(f)
	clf = a.load()
with open(filename1,'rb') as f:
	b = pickle.Unpickler(f)
	cv = b.load()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		message = request.form['text']
		data = [message]
		vect = cv.transform(data).toarray()
		sa_prediction = clf.predict(vect)
	return render_template('result.html',prediction = sa_prediction)

if __name__ == '__main__':
	app.run(debug=True)