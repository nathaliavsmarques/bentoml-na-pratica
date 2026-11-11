
import requests
import streamlit as st


# ============================================================
# CONFIGURAÇÕES
# ============================================================

API_URL = "http://localhost:3000/predict"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# CABEÇALHO
# ============================================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    ### Sistema de previsão de cancelamento de clientes

    Informe as características de um cliente para que o modelo
    de Machine Learning estime a probabilidade de cancelamento
    dos serviços contratados.

    **Modelo:** Random Forest | **API:** BentoML
    """
)

st.divider()


# ============================================================
# FORMULÁRIO
# ============================================================

with st.form("customer_form"):

    st.subheader("1. Perfil do cliente")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox(
            "Gênero",
            ["Male", "Female"],
            format_func=lambda x: {
                "Male": "Masculino",
                "Female": "Feminino",
            }[x],
        )

    with col2:
        senior_citizen = st.selectbox(
            "Idoso?",
            [0, 1],
            format_func=lambda x: "Sim" if x == 1 else "Não",
        )

    with col3:
        partner = st.selectbox(
            "Possui parceiro(a)?",
            ["Yes", "No"],
            format_func=lambda x: "Sim" if x == "Yes" else "Não",
        )

    with col4:
        dependents = st.selectbox(
            "Possui dependentes?",
            ["Yes", "No"],
            format_func=lambda x: "Sim" if x == "Yes" else "Não",
        )

    st.divider()

    st.subheader("2. Serviços contratados")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox(
            "Serviço telefônico",
            ["Yes", "No"],
            format_func=lambda x: "Sim" if x == "Yes" else "Não",
        )

        internet_service = st.selectbox(
            "Tipo de internet",
            ["DSL", "Fiber optic", "No"],
            format_func=lambda x: {
                "DSL": "DSL",
                "Fiber optic": "Fibra óptica",
                "No": "Sem internet",
            }[x],
        )

        online_security = st.selectbox(
            "Segurança online",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

        online_backup = st.selectbox(
            "Backup online",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

    with col2:
        multiple_lines = st.selectbox(
            "Múltiplas linhas telefônicas",
            ["Yes", "No", "No phone service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No phone service": "Sem serviço telefônico",
            }[x],
        )

        device_protection = st.selectbox(
            "Proteção de dispositivos",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

        tech_support = st.selectbox(
            "Suporte técnico",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming de TV",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

        streaming_movies = st.selectbox(
            "Streaming de filmes",
            ["Yes", "No", "No internet service"],
            format_func=lambda x: {
                "Yes": "Sim",
                "No": "Não",
                "No internet service": "Sem serviço de internet",
            }[x],
        )

    st.divider()

    st.subheader("3. Contrato e pagamento")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Tipo de contrato",
            ["Month-to-month", "One year", "Two year"],
            format_func=lambda x: {
                "Month-to-month": "Mensal",
                "One year": "Um ano",
                "Two year": "Dois anos",
            }[x],
        )

    with col2:
        paperless_billing = st.selectbox(
            "Fatura digital?",
            ["Yes", "No"],
            format_func=lambda x: "Sim" if x == "Yes" else "Não",
        )

    with col3:
        payment_method = st.selectbox(
            "Forma de pagamento",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            format_func=lambda x: {
                "Electronic check": "Cheque eletrônico",
                "Mailed check": "Cheque enviado pelo correio",
                "Bank transfer (automatic)": "Transferência automática",
                "Credit card (automatic)": "Cartão de crédito automático",
            }[x],
        )

    st.divider()

    st.subheader("4. Tempo de permanência e cobranças")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input(
            "Tempo como cliente (meses)",
            min_value=0,
            max_value=72,
            value=2,
            step=1,
        )

    with col2:
        monthly_charges = st.number_input(
            "Cobrança mensal (US$)",
            min_value=0.0,
            max_value=200.0,
            value=70.70,
            step=0.50,
        )

    with col3:
        total_charges = st.number_input(
            "Cobranças acumuladas (US$)",
            min_value=0.0,
            value=141.40,
            step=1.0,
        )

    st.caption(
        "Os valores monetários devem ser informados em dólares, "
        "conforme a unidade utilizada no dataset original."
    )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Prever risco de cancelamento",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ENVIO DOS DADOS PARA A API BENTOML
# ============================================================

if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        with st.spinner("Consultando modelo de Machine Learning..."):

            response = requests.post(
                API_URL,
                json={"input_data": customer_data},
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

        st.divider()
        st.subheader("5. Resultado da previsão")

        prediction = result["prediction"]
        probability = result["churn_probability"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probabilidade estimada de cancelamento",
                f"{probability * 100:.2f}%",
            )

            st.progress(probability)

        with col2:
            st.metric(
                "Classificação do modelo",
                "Cancelamento" if prediction == "Yes" else "Sem indicação de cancelamento",
            )

        if prediction == "Yes":
            st.warning(
                "⚠️ O modelo identificou indicação de cancelamento "
                "para este cliente."
            )
        else:
            st.success(
                "✅ O modelo não identificou indicação de "
                "cancelamento para este cliente."
            )

        st.info(
            "Esta é uma estimativa estatística produzida pelo "
            "modelo, não uma garantia de comportamento futuro."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Não foi possível conectar à API BentoML. "
            "Verifique se o servidor está em execução na porta 3000."
        )

    except requests.exceptions.Timeout:
        st.error("A API demorou muito para responder. Tente novamente.")

    except requests.exceptions.HTTPError as error:
        st.error(f"A API retornou um erro HTTP: {error}")

    except (KeyError, ValueError) as error:
        st.error(f"Não foi possível interpretar a resposta da API: {error}")

    except requests.exceptions.RequestException as error:
        st.error(f"Erro na comunicação com a API: {error}")


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Projeto acadêmico de MLOps | Customer Churn Prediction | "
    "Random Forest + BentoML + Streamlit"
)
