"""
preprocessing.py
Limpieza de texto, tokenización, stopwords, lematización (NLTK) y POS Tagging (spaCy).
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# ── Descarga de recursos NLTK (solo la primera vez) ──────────────────────────
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)


# =============================================================================
#  LIMPIEZA CON REGEX
# =============================================================================

def clean_text(text: str) -> str:
    """
    Elimina caracteres no alfabéticos, paréntesis y espacios múltiples.
    Retorna el texto en minúsculas.
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'[()]',      ' ', text)
    text = re.sub(r'\s+',       ' ', text)
    return text.lower().strip()


def apply_cleaning(df: pd.DataFrame, col: str = 'review') -> pd.DataFrame:
    """
    Aplica clean_text sobre la columna `col` y agrega 'review_clean'.
    También mapea la columna 'sentiment' a valores binarios en 'sentiment_map'.
    """
    df = df.copy()
    df['review_clean']   = df[col].apply(clean_text)
    df['sentiment_map']  = df['sentiment'].map({'Positive': 1, 'Negative': 0})
    return df


# =============================================================================
#  PIPELINE NLTK: tokenización → stopwords → lematización
# =============================================================================

_stop_words  = set(stopwords.words('english'))
_lemmatizer  = WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    """Tokeniza una cadena de texto."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Elimina stopwords de una lista de tokens."""
    return [w for w in tokens if w not in _stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lematiza una lista de tokens."""
    return [_lemmatizer.lemmatize(w) for w in tokens]


def nltk_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de NLTK sobre 'review_clean'.
    Devuelve un DataFrame con la columna 'review_lematizer' (lista de tokens)
    y 'review_lematizer_str' (string unido por espacios).
    """
    df = df.copy()
    df['review_tokenized']  = df['review_clean'].apply(tokenize)
    df['review_stopwords']  = df['review_tokenized'].apply(remove_stopwords)
    df['review_lematizer']  = df['review_stopwords'].apply(lemmatize)
    df['review_lematizer_str'] = df['review_lematizer'].apply(lambda x: ' '.join(x))

    # Nos quedamos solo con las columnas de interés
    return df.drop(columns=['review_tokenized', 'review_stopwords'])


# =============================================================================
#  PIPELINE spaCy: lematización contextual + POS Tagging
# =============================================================================

def spacy_lemmatize(df: pd.DataFrame,
                    col: str = 'review_clean',
                    batch_size: int = 500) -> list[str]:
    """
    Lematiza usando spaCy (sin NER ni parser para mayor velocidad).
    Retorna una lista de strings lematizados.
    """
    nlp = spacy.load('en_core_web_sm')
    result = []
    for doc in nlp.pipe(df[col], disable=['ner', 'parser'], batch_size=batch_size):
        result.append(' '.join([
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
        ]))
    return result


def pos_tagging(texts: list[str], batch_size: int = 500) -> pd.DataFrame:
    """
    Aplica POS Tagging con spaCy y filtra VERB, ADJ y NOUN (mayor carga semántica).
    Retorna un DataFrame con columnas: review_id, token, pos_tag.
    """
    nlp = spacy.load('en_core_web_sm')
    rows = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                rows.append({
                    'review_id': i,
                    'token':     token.text,
                    'pos_tag':   token.pos_
                })
    df = pd.DataFrame(rows)
    # Filtramos solo las categorías con mayor carga semántica
    return df[df['pos_tag'].isin(['VERB', 'ADJ', 'NOUN'])]


def build_features(concat_df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los tokens por review_id y hace merge con el DataFrame original
    para recuperar 'sentiment_map'. Maneja el posible desajuste de 1 registro
    usando merge() + notna().
    """
    features = pos_df.groupby('review_id')['token'].apply(' '.join).reset_index()

    concat_df = concat_df.reset_index(drop=True)
    concat_df.index.name = 'review_id'
    concat_df = concat_df.reset_index()

    result = concat_df.merge(features, on='review_id', how='left')
    result = result[result['token'].notna()].reset_index(drop=True)

    return result[['review_id', 'token', 'sentiment_map']].copy()
