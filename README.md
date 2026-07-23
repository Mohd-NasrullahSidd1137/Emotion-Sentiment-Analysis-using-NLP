# 😊 Emotion Sentiment Analysis using NLP

A Machine Learning and Natural Language Processing (NLP) based web application that predicts human emotions from text in real time. The application is built with **Python**, **Scikit-learn**, and **Streamlit**, and is deployed online for public use.

## 🌐 Live Demo

**Application:** https://emotion-sentiment-analysis-using-nlp.streamlit.app/

---

## 📌 Project Overview

This project classifies user-entered text into one of six emotions using a Machine Learning model.

Supported emotions:

* 😊 Joy
* 😢 Sadness
* 😡 Anger
* 😨 Fear
* ❤️ Love
* 😲 Surprise

The application provides:

* Real-time emotion prediction
* Confidence score
* Emotion probability distribution
* Interactive and responsive Streamlit interface

---

## 🚀 Features

* Real-time emotion detection
* Machine Learning-based prediction
* Natural Language Processing (NLP)
* Confidence score visualization
* Emotion probability table
* Interactive bar chart
* Modern Streamlit user interface
* Responsive design
* Fast prediction

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* Joblib
* Bag of Words (CountVectorizer)
* Logistic Regression

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Text Preprocessing
3. Feature Extraction using Bag of Words
4. Model Training with Logistic Regression
5. Model Evaluation
6. Model Serialization using Joblib
7. Streamlit Deployment

---

## 📊 Model Information

| Feature    | Value                          |
| ---------- | ------------------------------ |
| Algorithm  | Logistic Regression            |
| Vectorizer | Bag of Words (CountVectorizer) |
| Accuracy   | **88%**                        |
| Classes    | 6                              |

---

## 📂 Project Structure

```text
Emotion-Sentiment-Analysis/
│
├── app.py
├── emotion_model.pkl
├── bow_vectorizer.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/emotion-sentiment-analysis.git
```

Move into the project folder

```bash
cd emotion-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 Example Inputs

**Joy**

```
I got selected in my interview.
```

**Love**

```
I love spending time with my family.
```

**Sadness**

```
I lost my wallet yesterday.
```

**Fear**

```
I am scared of snakes.
```

**Anger**

```
I am very angry today.
```

**Surprise**

```
Wow! I can't believe this happened.
```

---

## 🎯 Future Improvements

* Deep Learning (LSTM / GRU)
* Transformer-based models (BERT)
* Speech-to-Emotion Detection
* Voice Input
* Multi-language Support
* Emotion History Dashboard
* API Integration

---

## 👨‍💻 Developer

**Mohd Nasrullah Siddiqui**

Data Analytics | Machine Learning | NLP | Python

GitHub: https://github.com/your-username

LinkedIn: https://linkedin.com/in/your-linkedin

---

## ⭐ If you found this project useful, don't forget to Star ⭐ the repository.
