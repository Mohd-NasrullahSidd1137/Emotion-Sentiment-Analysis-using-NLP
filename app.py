import streamlit as st
import joblib
import pandas as pd

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Emotion Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL
# ======================================================

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")

# ======================================================
# EMOTION LABELS
# ======================================================

emotion_map = {
    0: {
        "emoji": "😢",
        "name": "Sadness"
    },

    1: {
        "emoji": "😡",
        "name": "Anger"
    },

    2: {
        "emoji": "❤️",
        "name": "Love"
    },

    3: {
        "emoji": "😲",
        "name": "Surprise"
    },

    4: {
        "emoji": "😨",
        "name": "Fear"
    },

    5: {
        "emoji": "😊",
        "name": "Joy"
    }
}

# ======================================================
# PREMIUM CSS
# ======================================================

st.markdown("""
<style>

/* ===========================
BACKGROUND
=========================== */

.stApp{
    background:linear-gradient(135deg,#0F172A,#1E293B,#334155);
}

/* ===========================
REMOVE STREAMLIT HEADER
=========================== */

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

/* ===========================
MAIN TITLE
=========================== */

.title{
    text-align:center;
    font-size:52px;
    font-weight:bold;
    color:white;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#CBD5E1;
    margin-bottom:35px;
}

/* ===========================
MAIN PAGE TEXT
=========================== */

h1,h2,h3,h4,h5,h6{
    color:white !important;
}

p{
    color:white !important;
}

label{
    color:white !important;
}

li{
    color:white !important;
}

ul{
    color:white !important;
}

.stMarkdown{
    color:white !important;
}

/* ===========================
TEXT AREA
=========================== */

textarea{
    background:white !important;
    color:black !important;
    font-size:18px !important;
    border-radius:12px !important;
}

/* ===========================
BUTTON
=========================== */

.stButton>button{

    width:100%;

    height:55px;

    border-radius:12px;

    border:none;

    font-size:18px;

    font-weight:bold;

    background:#2563EB;

    color:white;

}

.stButton>button:hover{

    background:#1D4ED8;

    color:white;

}

/* ===========================
RESULT CARD
=========================== */

.result-box{

    background:white;

    border-radius:20px;

    padding:30px;

    text-align:center;

    box-shadow:0px 0px 20px rgba(0,0,0,.30);

    margin-top:25px;

}

.result-box h1{

    font-size:70px;

    color:black !important;

}

.result-box h2{

    color:#2563EB !important;

    font-size:36px;

}

.result-box p{

    color:black !important;

    font-size:18px;

}

/* ===========================
SIDEBAR
=========================== */

[data-testid="stSidebar"]{

    background:#F8FAFC;

}

/* Sidebar Text */

[data-testid="stSidebar"] *{

    color:black !important;

}

/* Sidebar Code */

[data-testid="stSidebar"] code{

    color:#2563EB !important;

}

/* ===========================
EXPANDER
=========================== */

.streamlit-expanderHeader{

    color:white !important;

    font-size:22px;

    font-weight:bold;

}

.streamlit-expanderContent{

    color:white !important;

}

/* ===========================
DATAFRAME
=========================== */

[data-testid="stDataFrame"]{

    background:white;

    color:black !important;

}

/* ===========================
METRIC
=========================== */

[data-testid="stMetricValue"]{

    color:white !important;

}

[data-testid="stMetricLabel"]{

    color:white !important;

}

/* ===========================
PROGRESS TEXT
=========================== */

[data-testid="stProgressBar"]{

    border-radius:10px;

}

/* ===========================
FOOTER
=========================== */

.footer{

    text-align:center;

    color:white;

    font-size:18px;

    margin-top:30px;

}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title("📌 Project Information")

    st.success("Emotion Sentiment Analysis")

    st.markdown("---")

    st.subheader("🤖 Model")

    st.write("**Algorithm:** Logistic Regression")

    st.write("**Vectorizer:** Bag of Words")

    st.write("**Accuracy:** 88%")

    st.write("**Classes:** 6")

    st.markdown("---")

    st.subheader("📝 Try These Examples")

    st.code("I got selected in my interview.")

    st.code("I love my parents.")

    st.code("I lost my wallet yesterday.")

    st.code("I am scared of snakes.")

    st.code("Wow! I can't believe this!")

    st.code("I am very angry today.")

    st.markdown("---")

    st.info("👨‍💻 Developed by Mohd Nasrullah Siddiqui")

# ======================================================
# HERO SECTION
# ======================================================

st.markdown("""
<div class="title">
😊 Emotion Sentiment Analysis
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Detect Human Emotions from Text using
<b>Machine Learning</b> &
<b>Natural Language Processing (NLP)</b>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# INPUT SECTION
# ======================================================

st.subheader("✍️ Enter Your Text")

user_text = st.text_area(
    label="",
    placeholder="Example: I got selected in my interview and I am feeling amazing!",
    height=180
)

col1, col2 = st.columns(2)

with col1:

    predict = st.button(
        "🚀 Predict Emotion",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🗑️ Clear Text",
        use_container_width=True
    )

if clear:
    st.rerun()

# ======================================================
# PREDICTION
# ======================================================

if predict:

    if user_text.strip() == "":

        st.warning("⚠️ Please enter some text first.")

    else:

        # Vectorize Input
        vector = vectorizer.transform([user_text])

        # Prediction
        prediction = int(model.predict(vector)[0])

        emoji = emotion_map[prediction]["emoji"]

        emotion = emotion_map[prediction]["name"]

        # Confidence
        probabilities = model.predict_proba(vector)[0]

        confidence = probabilities.max() * 100

        st.markdown("<br>", unsafe_allow_html=True)

        # ===================================================
        # RESULT CARD
        # ===================================================

        st.markdown(f"""
        <div class="result-box">

            <h1>{emoji}</h1>

            <h2>{emotion}</h2>

            <p>
            The model predicts that the entered text expresses
            <b>{emotion}</b>.
            </p>

        </div>

        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ===================================================
        # CONFIDENCE SCORE
        # ===================================================

        st.subheader("📊 Confidence Score")

        st.progress(float(confidence / 100))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                emotion
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col3:
            st.metric(
                "Model",
                "Logistic Regression"
            )

        # ======================================================
        # EMOTION PROBABILITIES
        # ======================================================

        st.markdown("---")

        st.subheader("📈 Emotion Probability")

        labels = [
            "😢 Sadness",
            "😡 Anger",
            "❤️ Love",
            "😲 Surprise",
            "😨 Fear",
            "😊 Joy"
        ]

        probability_df = pd.DataFrame({
            "Emotion": labels,
            "Probability (%)": (probabilities * 100).round(2)
        })

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        # ======================================================
        # BAR CHART
        # ======================================================

        st.subheader("📊 Emotion Distribution")

        chart_df = probability_df.set_index("Emotion")

        st.bar_chart(
            chart_df["Probability (%)"],
            use_container_width=True
        )

# ======================================================
# ABOUT PROJECT
# ======================================================

st.markdown("---")

with st.expander("📖 About This Project", expanded=False):

    st.markdown("""

### 🎯 Emotion Sentiment Analysis

This application predicts human emotions from text using
Natural Language Processing (NLP) and Machine Learning.

### 🛠 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Bag of Words
- Logistic Regression

### 😊 Supported Emotions

- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

### 📊 Model Accuracy

**88%**

### 👨‍💻 Developed By

Mohd Nasrullah Siddiqui

""")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
<div class="footer">

Made with ❤️ using Python, Streamlit & Machine Learning

<br><br>

<b>© 2026 Mohd Nasrullah Siddiqui</b>

</div>
""", unsafe_allow_html=True)
