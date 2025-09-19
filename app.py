import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === PAGE CONFIG & CSS ===
st.set_page_config(page_title="Analisis Sentimen Timnas Indonesia", layout="wide")

st.markdown("""
<style>
/* Body background hitam */
body {
    background-color: #111111;
    color: white;
}

/* Judul & subjudul */
h1, h2, h3, h4, h5, h6 {
    color: white;
    text-align: center;
}

/* Tombol */
.stButton>button {
    background-color: #b22234;
    color: white;
    font-weight: bold;
}

/* Dropdown */
.stSelectbox>div>div>div {
    color: #b22234;
    font-weight: bold;
}

/* Teks area */
.stTextArea>div>textarea {
    background-color: #222222;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# === LOAD DATASET ===
@st.cache_data
def load_data():
    df = pd.read_csv("Analisis_sentimen_timnas_sepakbola_indonesia_di_era_STY.csv")
    df['label'] = df['label'].map({'negatif': 0, 'positif': 1})
    return df

df = load_data()

# === LOAD MODEL ===
nb_model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

MODEL_PATH = "matthewaldhino/indobert-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
bert_model.eval()

# === FUNGSI PREDIKSI ===
def predict_nb(text):
    X = vectorizer.transform([text])
    pred = nb_model.predict(X)[0]
    prob = nb_model.predict_proba(X)[0][pred]
    return ("Positif" if pred == 1 else "Negatif"), float(prob)

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).cpu().item()
    return ("Positif" if pred == 1 else "Negatif"), float(probs[0][pred].cpu())

# === HEADER ===
st.markdown("<h1 style='font-size:36px; text-align:center;'>Analisis Sentimen Penggemar Timnas Indonesia Era Shin Tae-yong</h1>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# === TUJUAN ===
st.markdown("<h2 style='text-align:center;'>🎯 Tujuan Analisis</h2>", unsafe_allow_html=True)
st.write("""
Menganalisis kecenderungan sentimen penggemar terhadap performa Tim Nasional Indonesia 
selama periode kepelatihan Shin Tae-yong berdasarkan komentar berlabel positif dan negatif.
""")

# === DISTRIBUSI SENTIMEN ===
st.markdown("<h2 style='text-align:center;'>🔍 Distribusi Sentimen</h2>", unsafe_allow_html=True)
label_counts = df['label'].value_counts().rename({0: 'Negatif', 1: 'Positif'}).reset_index()
label_counts.columns = ['Sentimen', 'Jumlah']
fig_sentimen = px.bar(label_counts, x='Sentimen', y='Jumlah', color='Sentimen',
                      color_discrete_map={'Positif':'#b22234','Negatif':'#ffffff'},
                      text='Jumlah')
fig_sentimen.update_layout(
    showlegend=False,
    plot_bgcolor='#111111',  # background hitam
    paper_bgcolor='#111111',
    font=dict(color='white')
)
st.plotly_chart(fig_sentimen, use_container_width=True)

# === WORDCLOUD ===
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align:center;'>Wordcloud Komentar Positif</h3>", unsafe_allow_html=True)
    text_pos = " ".join(df[df['label'] == 1]['komentar'])
    wc_pos = WordCloud(width=500, height=300, background_color='#111111', colormap='Reds').generate(text_pos)
    plt.imshow(wc_pos)
    plt.axis("off")
    st.pyplot(plt)
with col2:
    st.markdown("<h3 style='text-align:center;'>Wordcloud Komentar Negatif</h3>", unsafe_allow_html=True)
    text_neg = " ".join(df[df['label'] == 0]['komentar'])
    wc_neg = WordCloud(width=500, height=300, background_color='#111111', colormap='gray').generate(text_neg)
    plt.imshow(wc_neg)
    plt.axis("off")
    st.pyplot(plt)

# === HASIL PEMODELAN ===
st.markdown("<h2 style='text-align:center;'>📈 Perbandingan Model</h2>", unsafe_allow_html=True)
metrics = pd.DataFrame({
    "Model": ["Naive Bayes", "IndoBERT"],
    "Akurasi": [0.675, 0.7375],
    "F1-Score": [0.67, 0.74]
})
fig_model = px.bar(metrics.melt(id_vars='Model', value_vars=['Akurasi','F1-Score']),
                   x='Model', y='value', color='variable', barmode='group',
                   color_discrete_map={'Akurasi':'#b22234','F1-Score':'#ffffff'})
fig_model.update_layout(
    plot_bgcolor='#111111',
    paper_bgcolor='#111111',
    font=dict(color='white')
)
st.plotly_chart(fig_model, use_container_width=True)

# === PREDIKSI ===
st.markdown("<h2 style='text-align:center;'>💡 Prediksi Sentimen Komentar</h2>", unsafe_allow_html=True)
model_choice = st.selectbox("Pilih Model:", ["Naive Bayes", "IndoBERT"])
user_input = st.text_area("Masukkan komentar di sini:")

if st.button("Prediksi"):
    if user_input.strip():
        if model_choice == "Naive Bayes":
            hasil, prob = predict_nb(user_input)
        else:
            hasil, prob = predict_bert(user_input)
        st.success(f"Hasil Prediksi: **{hasil}** (confidence: {prob:.2f})")
    else:
        st.warning("Masukkan komentar terlebih dahulu.")

# === KESIMPULAN ===
st.markdown("<h2 style='text-align:center;'>📝 Kesimpulan</h2>", unsafe_allow_html=True)
st.write("""
Model **IndoBERT** yang di-fine-tuning mampu meningkatkan akurasi prediksi sentimen hingga **73.7%**,
mengungguli model tradisional **Naive Bayes**. 
Penggunaan model berbasis Transformer dapat memahami konteks komentar
dengan lebih baik dan menghasilkan analisis sentimen yang lebih akurat.

""")

