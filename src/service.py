
import pandas as pd
import bentoml

from typing import Any


# Referência ao modelo registrado no BentoML Model Store.
MODEL_TAG = "churn_pipeline:iqpnzng572bm2aav"


# Campos esperados pelo modelo.
REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@bentoml.service(
    name="churn_prediction",
    resources={"cpu": "1"},
)
class ChurnService:

    # Declara o modelo como dependência do serviço.
    model_ref = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self):
        # Carrega o modelo registrado no Model Store.
        self.model = self.model_ref.load_model()

        print("Modelo Customer Churn carregado com sucesso!")

    @bentoml.api
    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:

        # Verifica se todos os campos necessários foram enviados.
        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in input_data
        ]

        if missing_features:
            raise ValueError(
                f"Campos obrigatórios ausentes: {missing_features}"
            )

        # Organiza os dados no formato esperado pelo pipeline.
        customer_data = pd.DataFrame(
            [input_data],
            columns=REQUIRED_FEATURES,
        )

        # Executa a previsão.
        prediction = self.model.predict(customer_data)[0]

        # Obtém as probabilidades das duas classes.
        probabilities = self.model.predict_proba(customer_data)[0]

        classes = list(self.model.classes_)

        churn_index = classes.index("Yes")
        churn_probability = float(probabilities[churn_index])

        # Monta a resposta da API.
        return {
            "prediction": str(prediction),
            "churn_probability": round(churn_probability, 4),
            "message": (
                "Cliente com indicação de cancelamento."
                if prediction == "Yes"
                else "Cliente sem indicação de cancelamento."
            ),
        }
