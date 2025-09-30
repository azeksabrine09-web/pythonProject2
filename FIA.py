import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Config Streamlit
# -------------------------
st.set_page_config(page_title="Financial Inclusion", layout="wide")
st.title("🌍 Prédiction de l’inclusion financière")

# -------------------------
# Chargement et préparation des données
# -------------------------
@st.cache_data
def load_and_train():
    # Charger le dataset
    df = pd.read_csv("Financial_inclusion_dataset.csv")

    # Définir la cible
    TARGET_COL = "bank_account"

    # Séparer X et y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identifier colonnes numériques / catégorielles
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Encoder les colonnes catégorielles
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entraîner modèle
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluer
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, df, X.columns.tolist(), num_cols, cat_cols, acc


# -------------------------
# Lancer le modèle
# -------------------------
model, df, all_features, num_cols, cat_cols, acc = load_and_train()

st.success(f"✅ Modèle entraîné avec une précision de **{acc:.2%}**")

# -------------------------
# Formulaire de prédiction
# -------------------------
st.header("🔮 Simulation d’un individu")

input_data = {}
for col in num_cols:
    input_data[col] = st.number_input(f"{col}", value=0)

for col in cat_cols:
    input_data[col] = st.selectbox(f"{col}", options=df[col].unique())

# Transformer en DataFrame
input_df = pd.DataFrame([input_data])

# Encoder les colonnes catégorielles pour correspondre au modèle
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Ré-aligner avec colonnes du modèle
input_df = input_df.reindex(columns=all_features, fill_value=0)

# Prédiction
if st.button("Prédire"):
    pred = model.predict(input_df)[0]
    st.write("Résultat :", "✅ Possède un compte bancaire" if pred == "Yes" else "❌ Pas de compte bancaire")
