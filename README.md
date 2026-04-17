# 🎬 Análisis de Sentimientos - Reseñas IMDb

Proyecto de NLP para clasificación binaria de sentimientos (positivo/negativo) sobre reseñas de películas del dataset IMDb.

Realizado por **Lautaro Rodriguez**.

---

## 📁 Estructura del repositorio

```
nlp-sentiment-imdb/
├── notebooks/
│   └── Analisis_nlp_Lautaro_Rodriguez.ipynb   # Notebook original exploratorio
├── src/
│   ├── preprocessing.py      # Limpieza, NLTK pipeline, spaCy POS Tagging
│   ├── visualization.py      # WordCloud, N-gramas, barras, matrices de confusión
│   └── models/
│       ├── classical.py      # Logistic Regression + SVC con HalvingGridSearchCV
│       ├── rnn_models.py     # RNN, LSTM, GRU con TensorFlow/Keras
│       └── pytorch_model.py  # TextClassifier + EarlyStopping con PyTorch
├── main.py                   # Pipeline principal (ejecuta todo)
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/nlp-sentiment-imdb.git
cd nlp-sentiment-imdb

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo de spaCy
python -m spacy download en_core_web_sm
```

---

## ▶️ Ejecución

### Pipeline completo
```bash
python main.py
```

### Usar módulos por separado
```python
from src.preprocessing import apply_cleaning, nltk_pipeline
from src.models.classical import train_logistic_regression
from src.visualization import plot_wordclouds_by_sentiment

# Cargar y preprocesar
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/laurenBack98/Data_set_sentiments_2/refs/heads/main/Review.csv")
df_clean = apply_cleaning(df)
df_nltk  = nltk_pipeline(df_clean)

# Entrenar un modelo
# ...
```

---

## 📊 Dataset

- **Fuente:** [Kaggle - IMDb Movie Review NLP](https://www.kaggle.com/datasets/krystalliu152/imbd-movie-reviewnpl)
- **Columnas:** `review` (texto), `sentiment` (Positive/Negative)
- **Clases:** Balanceadas (~50% positivo, ~50% negativo)

---

## 🧠 Modelos implementados

| Modelo               | Framework      | Descripción                                      |
|----------------------|----------------|--------------------------------------------------|
| Logistic Regression  | scikit-learn   | TF-IDF + LR con HalvingGridSearchCV              |
| SVC                  | scikit-learn   | TF-IDF + SVM con kernel lineal/rbf               |
| RNN                  | TensorFlow     | Red recurrente simple con Embedding              |
| LSTM                 | TensorFlow     | Memoria a largo y corto plazo                    |
| GRU                  | TensorFlow     | Unidad recurrente con compuertas (más eficiente) |
| TextClassifier       | PyTorch        | Red densa + BatchNorm + Dropout + EarlyStopping  |

---

## 📈 Resultados

Los modelos clásicos (SVC y Logistic Regression) superaron a las redes neuronales en este dataset de tamaño mediano, alcanzando un **F1-Score de ~0.86**.

| Modelo              | Accuracy | F1-Score |
|---------------------|----------|----------|
| SVC                 | ~0.87    | ~0.86    |
| Logistic Regression | ~0.86    | ~0.86    |
| GRU                 | ~0.76    | ~0.76    |
| LSTM                | ~0.76    | ~0.76    |
| RNN                 | ~0.50    | ~0.50    |

---

## 🔮 Mejoras propuestas

- Incorporar embeddings preentrenados (**GloVe** o **BERT**) en la capa de entrada de las redes neuronales.
- Probar con un dataset más grande para que las redes neuronales puedan superar a los modelos clásicos.
- Implementar fine-tuning de un modelo transformador (ej. `bert-base-uncased`).

---

## 🛠️ Stack tecnológico

- **Python 3.10+**
- pandas, numpy, scikit-learn
- NLTK, spaCy (`en_core_web_sm`)
- TensorFlow / Keras
- PyTorch
- matplotlib, seaborn, wordcloud

