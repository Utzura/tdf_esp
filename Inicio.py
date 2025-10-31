import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---- CONFIGURACIÓN ----
st.set_page_config(page_title="TF-IDF en Español", page_icon="🔮", layout="wide")

# ---- SWITCH DE TEMA ----
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

col_toggle, col_title = st.columns([1, 8])
with col_toggle:
    dark_mode = st.toggle("🌙 Modo oscuro", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode

# ---- ESTILOS ----
if st.session_state.dark_mode:
    bg_color = "#0f0f10"
    text_color = "#e6e6e6"
    accent = "#c29fff"
    box_color = "#1a1a1b"
else:
    bg_color = "#f9f9fb"
    text_color = "#222"
    accent = "#7a3ef5"
    box_color = "#ffffff"

st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        color: {accent};
    }}
    textarea, input {{
        background-color: {box_color} !important;
        color: {text_color} !important;
        border: 1px solid #555 !important;
        border-radius: 8px !important;
    }}
    .stButton button {{
        background: linear-gradient(135deg, {accent}, #9b6cff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        transition: 0.3s;
    }}
    .stButton button:hover {{
        background: linear-gradient(135deg, #b68cff, #d2b8ff);
        transform: scale(1.03);
    }}
    .stSuccess, .stInfo, .stWarning {{
        background-color: {box_color} !important;
        color: {text_color} !important;
        border-left: 3px solid {accent} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ---- TÍTULO ----
st.title("🔮 Analizador TF-IDF Interactivo")

# ---- DOCUMENTOS BASE ----
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ---- INTERFAZ ----
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📜 Documentos (uno por línea):", default_docs, height=180)
    question = st.text_input("💬 Tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    sugeridas = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]
    for p in sugeridas:
        if st.button(p, use_container_width=True):
            st.session_state.question = p
            st.rerun()

if "question" in st.session_state:
    question = st.session_state.question

# ---- ANÁLISIS ----
if st.button("✨ Analizar texto", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)

        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 🎯 Resultado del análisis")
        st.markdown(f"**Pregunta:** {question}")

        if best_score > 0.01:
            st.success(f"**Respuesta más similar:** {best_doc}")
            st.info(f"📈 Grado de similitud: `{best_score:.3f}`")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Grado de similitud: `{best_score:.3f}`")
