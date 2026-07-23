import streamlit as st
import joblib
import pandas as pd

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Emotion Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# ----------------------------------
# Load Model
# ----------------------------------

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")

# ----------------------------------
# Emotion Mapping
# ----------------------------------

emotion_map = {
    0: ("😢", "Sadness"),
    1: ("😡", "Anger"),
    2: ("❤️", "Love"),
    3: ("😲", "Surprise"),
    4: ("😨", "Fear"),
    5: ("😊", "Joy")
}

# ----------------------------------
# Custom CSS
# ----------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

.title{
text-align:center;
font-size:44px;
font-weight:bold;
color:white;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cbd5e1;
margin-bottom:30px;
}

.result-box{
background:white;
padding:25px;
border-radius:18px;
text-align:center;
box-shadow:0px 5px 20px rgba(0,0,0,.3);
margin-top:20px;
}

.result-box h1{
color:#1e293b;
font-size:40px;
}

.result-box h2{
color:#2563eb;
}

.footer{
text-align:center;
color:white;
margin-top:40px;
font-size:16px;
}

.stButton>button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
border-radius:12px;
background:linear-gradient(to right,#2563eb,#06b6d4);
color:white;
border:none;
}

.stButton>button:hover{
background:linear-gradient(to right,#1d4ed8,#0891b2);
color:white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.success("Emotion Sentiment Analysis")

st.sidebar.write("### Model Details")

st.sidebar.write("""
- **Algorithm:** Logistic Regression
- **Vectorizer:** Bag of Words
- **Accuracy:** 88%
- **Classes:** 6
""")

st.sidebar.markdown("---")

st.sidebar.write("### Try These Examples")

st.sidebar.code("I got selected in my interview.")
st.sidebar.code("I love my parents.")
st.sidebar.code("I am very angry.")
st.sidebar.code("I lost my wallet.")
st.sidebar.code("I am scared of snakes.")
st.sidebar.code("Wow! I can't believe it!")

# ----------------------------------
# Title
# ----------------------------------

st.markdown("<div class='title'>😊 Emotion Sentiment Analysis</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Machine Learning + NLP + Streamlit</div>", unsafe_allow_html=True)

# ----------------------------------
# Input
# ----------------------------------

user_text = st.text_area(
    "✍ Enter your sentence",
    height=180,
    placeholder="Example: I am very happy today because I got selected."
)

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("🚀 Predict Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        vector = vectorizer.transform([user_text])

        prediction = model.predict(vector)[0]

        emoji, emotion = emotion_map[int(prediction)]

        st.markdown(f"""
        <div class='result-box'>
        <h1>{emoji}</h1>
        <h2>{emotion}</h2>
        <p><b>The model predicts that the given text expresses <span style='color:#2563eb'>{emotion}</span>.</b></p>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(vector)[0]

            confidence = probs.max() * 100

            st.write("## 📊 Confidence Score")

            st.progress(int(confidence))

            st.success(f"{confidence:.2f}%")

            st.write("## 📈 Emotion Probabilities")

            labels = [
                "😢 Sadness",
                "😡 Anger",
                "❤️ Love",
                "😲 Surprise",
                "😨 Fear",
                "😊 Joy"
            ]

            df = pd.DataFrame({
                "Emotion": labels,
                "Probability (%)": (probs * 100).round(2)
            })

            st.dataframe(df, use_container_width=True)

# ----------------------------------
# About Project
# ----------------------------------

with st.expander("📖 About This Project"):

    st.write("""
### Emotion Sentiment Analysis using NLP

This project predicts emotions from text using
Natural Language Processing (NLP).

### Tech Stack

- Python
- Scikit-Learn
- Bag of Words
- Logistic Regression
- Streamlit

### Supported Emotions

- 😊 Joy
- 😢 Sadness
- 😡 Anger
- 😨 Fear
- ❤️ Love
- 😲 Surprise

### Model Accuracy

**88%**
""")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("""
<hr>
<div class='footer'>

Made with ❤️ using Streamlit

<br><br>

<b>Developed by Mohd Nasrullah Siddiqui</b>

</div>
""", unsafe_allow_html=True)
