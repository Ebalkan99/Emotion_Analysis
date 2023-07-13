# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:53:07 2023
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import numpy as np

app = Flask(__name__)

# TfidfVectorizer ve SVC modelini yükleyin
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Eğitimde probability=False olarak ayarlanmış bir modeli yükleyin
model = pickle.load(open("model.pkl", "rb"))
model.decision_function_shape = 'ovr'

# Sınıf indekslerini etiketlere eşleştirmek için bir sözlük oluşturun
label_mapping = {0: "negatif", 1: "pozitif", 2: "nötr", 3: "korku", 4: "mutluluk", 5: "hüzün"}

@app.route('/')
def home():
    return render_template('index.html', sentiment="")

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    
    # Metni TfidfVectorizer'a göre vektöre dönüştürün
    vectorized_text = vectorizer.transform([text])
    
    # Duygu analizini yapmak için SVC modelini kullanın
    decision_values = model.decision_function(vectorized_text)
    decision_values = np.squeeze(decision_values)
    positive_decision_values = decision_values - np.min(decision_values)
    probabilities = positive_decision_values / np.sum(positive_decision_values)
    results = {label_mapping[i]: round(probability * 100, 2) for i, probability in enumerate(probabilities)}
   
    # Hatalı değerle ilgili işlemler yapabilirsiniz.

    return render_template('index.html', sentiment=results)


if __name__ == '__main__':
    app.run(debug=True)

