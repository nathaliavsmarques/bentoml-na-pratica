
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

RANDOM_STATE = 42


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

print("\nCarregando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dimensões originais: {df.shape}")

# Converte TotalCharges para numérico.
# Valores vazios serão convertidos em NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Os 11 registros ausentes foram identificados na EDA.
# Todos possuem tenure = 0, portanto TotalCharges = 0.
mask = df["TotalCharges"].isna() & (df["tenure"] == 0)

df.loc[mask, "TotalCharges"] = 0

# Verificação para evitar tratamento silencioso de outros ausentes.
if df["TotalCharges"].isna().any():
    raise ValueError(
        "Existem valores ausentes em TotalCharges "
        "que precisam ser investigados."
    )

# Remove o identificador do cliente.
df = df.drop(columns=["customerID"])

# Separa variáveis preditoras e variável-alvo.
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Total de registros: {len(df)}")
print(f"Distribuição da variável-alvo:\n{y.value_counts()}")


# ============================================================
# 3. IDENTIFICAÇÃO DAS VARIÁVEIS
# ============================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]

print("\nVariáveis numéricas:", numeric_features)
print("Quantidade de variáveis categóricas:", len(categorical_features))


# ============================================================
# 4. PRÉ-PROCESSAMENTO
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 5. DIVISÃO DOS DADOS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nDivisão dos dados:")
print(f"Treinamento: {X_train.shape[0]} registros")
print(f"Teste: {X_test.shape[0]} registros")


# ============================================================
# 6. MODELO DE REFERÊNCIA (DUMMY CLASSIFIER)
# ============================================================

print("\nTreinando modelo de referência...")

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_test)


# ============================================================
# 7. RANDOM FOREST + PIPELINE
# ============================================================

print("\nTreinando Random Forest...")

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)

# Localiza a probabilidade correspondente à classe Yes.
positive_class_index = list(
    model_pipeline.named_steps["classifier"].classes_
).index("Yes")

churn_probabilities = probabilities[:, positive_class_index]


# ============================================================
# 8. AVALIAÇÃO DOS MODELOS
# ============================================================

def evaluate_model(name, y_true, y_pred):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_true, y_pred, pos_label='Yes', zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_true, y_pred, pos_label='Yes', zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(y_true, y_pred, pos_label='Yes', zero_division=0):.4f}"
    )

    print("\nRelatório de classificação:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Matriz de confusão:")
    print(confusion_matrix(y_true, y_pred, labels=["No", "Yes"]))


evaluate_model(
    "MODELO DE REFERÊNCIA - DUMMY CLASSIFIER",
    y_test,
    dummy_predictions,
)

evaluate_model(
    "MODELO PRINCIPAL - RANDOM FOREST",
    y_test,
    predictions,
)

roc_auc = roc_auc_score(
    (y_test == "Yes").astype(int),
    churn_probabilities,
)

print(f"\nROC-AUC do Random Forest: {roc_auc:.4f}")


# ============================================================
# 9. SALVAMENTO DO MODELO
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model_pipeline, MODEL_PATH)

print("\nModelo salvo com sucesso!")
print(f"Localização: {MODEL_PATH}")
