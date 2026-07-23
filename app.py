import streamlit as st
import joblib
import pandas as pd

# ===========================================================
# PAGE CONFIGURATION
# ===========================================================

st.set_page_config(
    page_title="Emotion Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================
# LOAD MODEL
# ===========================================================

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")

# ===========================================================
# EMOTION LABELS
# ===========================================================

emotion_map = {
    0: ("😢", "Sadness"),
    1: ("😡", "Anger"),
    2: ("❤️", "Love"),
    3: ("😲", "Surprise"),
    4: ("😨", "Fear"),
    5: ("😊", "Joy")
}

st.markdown("""
<style>

/* ===========================
BACKGROUND
=========================== */

.stApp{
background:linear-gradient(135deg,#0F172A,#1E293B,#334155);
}

/* ===========================
TITLE
=========================== */

/* Sidebar Text */

[data-testid="stSidebar"] * {
    color: black !important;
}

/* Expander Text */

.streamlit-expanderHeader {
    color: white !important;
    font-weight: bold;
}

.streamlit-expanderContent {
    color: white !important;
}

/* Markdown Text */

.stMarkdown {
    color: white !important;
}

/* Lists */

ul, li {
    color: white !important;
}

/* Table */

table, th, td {
    color: white !important;
}

/* DataFrame */

[data-testid="stDataFrame"] {
    color: black !important;
}

.title{
text-align:center;
font-size:55px;
font-weight:bold;
color:white;
margin-top:10px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#CBD5E1;
margin-bottom:30px;
}

/* ===========================
RESULT CARD
=========================== */

.result-box{

background:white;

padding:30px;

border-radius:20px;

box-shadow:0px 8px 25px rgba(0,0,0,.35);

text-align:center;

margin-top:25px;

}

.result-box h1{

font-size:60px;

}

.result-box h2{

font-size:34px;

color:#2563EB;

}

/* ===========================
BUTTON
=========================== */

.stButton>button{

width:100%;

height:55px;

font-size:20px;

font-weight:bold;

background:#2563EB;

color:white;

border-radius:12px;

border:none;

}

.stButton>button:hover{

background:#1D4ED8;

color:white;

}

/* ===========================
TEXT AREA
=========================== */

textarea{

font-size:18px !important;

}

/* ===========================
HEADINGS
=========================== */

h1,h2,h3,h4,h5,h6,label,p{

color:white !important;

}

/* ===========================
SIDEBAR
=========================== */

[data-testid="stSidebar"]{

background:#F8FAFC;

}

</style>
""",unsafe_allow_html=True)

# ===========================================================
# SIDEBAR
# ===========================================================

st.sidebar.title("📌 Project Information")

st.sidebar.success("Emotion Sentiment Analysis")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model")

st.sidebar.write("**Algorithm:** Logistic Regression")

st.sidebar.write("**Vectorizer:** Bag of Words")

st.sidebar.write("**Accuracy:** 88%")

st.sidebar.write("**Classes:** 6")

st.sidebar.markdown("---")

st.sidebar.subheader("📝 Try These Examples")

st.sidebar.code("I got selected in my interview.")

st.sidebar.code("I love my parents.")

st.sidebar.code("I lost my wallet yesterday.")

st.sidebar.code("I am scared of snakes.")

st.sidebar.code("Wow! I can't believe this!")

st.sidebar.code("I am very angry today.")

st.sidebar.markdown("---")

st.sidebar.info("Developed by Mohd Nasrullah Siddiqui")

# ===========================================================
# MAIN TITLE
# ===========================================================

st.markdown("""
<div class='title'>
😊 Emotion Sentiment Analysis
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Detect Human Emotions using <b>Machine Learning</b>, <b>Natural Language Processing (NLP)</b> and <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================================================
# INPUT SECTION
# ===========================================================

st.subheader("✍️ Enter Your Text")

user_text = st.text_area(
    "",
    height=180,
    placeholder="Example: I am very happy because I got selected in my interview."
)

col1, col2 = st.columns([1,1])

with col1:

    predict = st.button(
        "🚀 Predict Emotion",
        use_container_width=True
    )

with col2:

    if st.button(
        "🗑️ Clear Text",
        use_container_width=True
    ):
        st.rerun()

# ===========================================================
# PREDICTION
# ===========================================================

if predict:

    if user_text.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        vector = vectorizer.transform([user_text])

        prediction = model.predict(vector)[0]

        emoji, emotion = emotion_map[int(prediction)]

        st.markdown(f"""
        <div class='result-box'>

        <h1>{emoji}</h1>

        <h2>{emotion}</h2>

        <p style="font-size:18px;color:black;">
        The model predicts that the given text expresses
        <b>{emotion}</b>.
        </p>

        </div>

        """, unsafe_allow_html=True)

        # =======================================
        # CONFIDENCE SCORE
        # =======================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(vector)[0]

            confidence = probabilities.max() * 100

            st.markdown("## 📊 Confidence Score")

            st.progress(int(confidence))

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

        # ==========================================
        # EMOTION PROBABILITIES
        # ==========================================

        st.markdown("## 📈 Emotion Probability")

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
            "Probability (%)": (probabilities * 100).round(2)
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("## 📊 Emotion Distribution")

        chart_df = pd.DataFrame({
            "Emotion": labels,
            "Probability": probabilities
        })

        st.bar_chart(
            chart_df,
            x="Emotion",
            y="Probability",
            use_container_width=True
        )

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.markdown("---")

with st.expander("📖 About This Project"):

    st.markdown("""

### 🎯 Emotion Sentiment Analysis using NLP

This project predicts human emotions from text using
Natural Language Processing.

### 🚀 Technologies Used

- Python
- NLP
- Bag of Words
- Logistic Regression
- Streamlit

### 😊 Supported Emotions

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

### 🎯 Model Accuracy

**88%**

""")

st.markdown("---")

st.markdown("""

<div style='text-align:center;
font-size:18px;
color:white;'>

Made with ❤️ using Streamlit

<br><br>

<b>Developed by Mohd Nasrullah Siddiqui</b>

</div>

""", unsafe_allow_html=True)
