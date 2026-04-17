"""
main.py
Pipeline principal del proyecto de análisis de sentimientos IMDb.
Ejecuta todos los pasos en orden: carga → preprocesamiento → EDA → modelado → evaluación.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from src.preprocessing import (
    apply_cleaning, nltk_pipeline,
    spacy_lemmatize, pos_tagging, build_features,
)
from src.visualization import (
    plot_wordclouds_by_sentiment, plot_ngram_wordcloud,
    plot_top_words, plot_history,
    plot_confusion_matrices, metrics_table,
)
from src.models.classical  import train_logistic_regression, train_svc
from src.models.rnn_models  import prepare_sequences, build_rnn, build_lstm, build_gru, train_rnn_model
from src.models.pytorch_model import train_pytorch_model


# =============================================================================
#  1. CARGA DE DATOS
# =============================================================================

DATA_URL = (
    "https://raw.githubusercontent.com/laurenBack98/"
    "Data_set_sentiments_2/refs/heads/main/Review.csv"
)

print("=" * 60)
print("1. Cargando datos...")
print("=" * 60)
df = pd.read_csv(DATA_URL)
print(df.head(10))
print(df.isnull().sum())
print(df.info())
print(df['sentiment'].value_counts())


# =============================================================================
#  2. PREPROCESAMIENTO
# =============================================================================

print("\n" + "=" * 60)
print("2. Preprocesamiento...")
print("=" * 60)

# Limpieza + mapeo de sentimiento
df_clean = apply_cleaning(df)

# Pipeline NLTK: tokenización, stopwords, lematización
df_nltk = nltk_pipeline(df_clean)

# Pipeline spaCy: lematización contextual
spacy_lemmas = spacy_lemmatize(df_clean)
df_clean['review_final'] = spacy_lemmas

import pandas as pd
concat_df = pd.concat([df_clean, pd.DataFrame({'review_final': spacy_lemmas})], axis=1)

# POS Tagging (VERB, ADJ, NOUN)
pos_df   = pos_tagging(concat_df['review_final'].tolist())
features = build_features(concat_df, pos_df)
print(f"Features shape: {features.shape}")


# =============================================================================
#  3. VISUALIZACIÓN / EDA
# =============================================================================

print("\n" + "=" * 60)
print("3. Visualizaciones EDA...")
print("=" * 60)

plot_wordclouds_by_sentiment(df_nltk)

from nltk.corpus import stopwords
stop_w = list(stopwords.words('english'))

for val, label in [(None, 'General'), (1, 'Positivos'), (0, 'Negativos')]:
    plot_ngram_wordcloud(df_nltk, sentiment_value=val,
                         stop_words=stop_w,
                         title=f'N-Gramas ({label})')

plot_top_words(df_nltk)


# =============================================================================
#  4. MODELOS CLÁSICOS DE ML
# =============================================================================

print("\n" + "=" * 60)
print("4. Modelos clásicos de ML...")
print("=" * 60)

model_lr,  x_test_lr,  y_test_lr,  y_pred_lr  = train_logistic_regression(features)
model_svc, x_test_svc, y_test_svc, y_pred_svc = train_svc(features)


# =============================================================================
#  5. REDES NEURONALES RECURRENTES (Keras)
# =============================================================================

print("\n" + "=" * 60)
print("5. Redes Neuronales Recurrentes (TF/Keras)...")
print("=" * 60)

MAX_WORDS = 10_000
MAX_LEN   = 100

X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn, _ = prepare_sequences(
    df_nltk, max_words=MAX_WORDS, max_len=MAX_LEN
)

model_rnn  = build_rnn(MAX_WORDS, MAX_LEN)
history_rnn, loss_rnn, acc_rnn = train_rnn_model(model_rnn, X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn)
plot_history(history_rnn, 'RNN')

model_lstm = build_lstm(MAX_WORDS, MAX_LEN)
history_lstm, loss_lstm, acc_lstm = train_rnn_model(model_lstm, X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn)
plot_history(history_lstm, 'LSTM')

model_gru  = build_gru(MAX_WORDS, MAX_LEN)
history_gru, loss_gru, acc_gru = train_rnn_model(model_gru, X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn)
plot_history(history_gru, 'GRU')


# =============================================================================
#  6. TEXTCLASSIFIER PYTORCH
# =============================================================================

print("\n" + "=" * 60)
print("6. TextClassifier PyTorch + Early Stopping...")
print("=" * 60)

_, y_val_pt, y_pred_pt = train_pytorch_model(features)


# =============================================================================
#  7. EVALUACIÓN COMPARATIVA
# =============================================================================

print("\n" + "=" * 60)
print("7. Evaluación comparativa de todos los modelos...")
print("=" * 60)

import numpy as np

y_pred_rnn_bin  = (model_rnn.predict(X_test_rnn)  > 0.5).astype(int)
y_pred_lstm_bin = (model_lstm.predict(X_test_rnn) > 0.5).astype(int)
y_pred_gru_bin  = (model_gru.predict(X_test_rnn)  > 0.5).astype(int)

models_results = [
    ('Logistic Regression', y_test_lr,  y_pred_lr),
    ('SVC',                 y_test_svc, y_pred_svc),
    ('RNN',                 y_test_rnn, y_pred_rnn_bin),
    ('LSTM',                y_test_rnn, y_pred_lstm_bin),
    ('GRU',                 y_test_rnn, y_pred_gru_bin),
]

plot_confusion_matrices(models_results)

df_metrics = metrics_table(models_results)
print("\nTabla de métricas comparativa:")
print(df_metrics.to_string(index=False))
