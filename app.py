import streamlit as st
import joblib

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
background:linear-gradient(135deg,#141E30,#243B55);
}

.title{
font-size:42px;
font-weight:bold;
color:white;
text-align:center;
}

.subtitle{
font-size:18px;
color:#dddddd;
text-align:center;
margin-bottom:25px;
}

.result-box{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:#0f172a;
box-shadow:0px 0px 15px rgba(0,0,0,0.3);
margin-top:20px;
}

.footer{
text-align:center;
color:white;
margin-top:40px;
}

.stButton>button{
background:#00B4DB;
background-image:linear-gradient(to right,#0083B0,#00B4DB);
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
height:50px;
width:100%;
border:none;
}

.stButton>button:hover{
background:#0077a6;
color:white;
}

textarea{
font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)

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

        st.markdown(
            f"<div class='result-box'>{emoji}<br>{emotion}</div>",
            unsafe_allow_html=True
        )

        if hasattr(model, "predict_proba"):

            confidence = model.predict_proba(vector).max() * 100

            st.write("### Confidence Score")

            st.progress(int(confidence))

            st.success(f"{confidence:.2f}%")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown(
"<div class='footer'>Developed by Mohd Nasrullah Siddiqui ❤️</div>",
unsafe_allow_html=True
)