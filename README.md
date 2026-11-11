
# BentoML na prática: empacotamento e disponibilização de modelos de Machine Learning

Projeto acadêmico desenvolvido para apresentar e demonstrar a utilização do **BentoML** como ferramenta de MLOps, explorando seus recursos para registro, organização, empacotamento e disponibilização de modelos de Machine Learning.

Como aplicação demonstrativa, foi desenvolvido um sistema de previsão de cancelamento de clientes (*Customer Churn Prediction*), utilizando um modelo Random Forest treinado com Scikit-learn.

**O foco deste trabalho é o BentoML e seu fluxo de utilização.** O modelo de Machine Learning é um exemplo prático para demonstrar a ferramenta e poderia ser substituído por qualquer outro modelo compatível.

## 1. Sobre o BentoML

O [BentoML](https://www.bentoml.com/) é uma ferramenta de código aberto que permite transformar modelos de Machine Learning em serviços de inferência prontos para execução e implantação.

Ele oferece recursos para organizar modelos, implementar serviços, empacotar suas dependências e gerar artefatos que podem ser executados em diferentes ambientes.

Neste projeto, são demonstradas as seguintes funcionalidades:

- **Model Store:** registro e gerenciamento de modelos de Machine Learning.
- **Bento Service:** implementação de um serviço que utiliza o modelo para realizar inferências.
- **Bento:** empacotamento do serviço, modelo e dependências necessárias.
- **Containerização:** geração de imagens Docker a partir dos Bentos.
- **API de inferência:** disponibilização do modelo por meio de endpoints HTTP.

## 2. Objetivo da demonstração

Demonstrar, na prática, como utilizar o BentoML para disponibilizar um modelo de Machine Learning como serviço.

O fluxo desenvolvido contempla:

1. Preparação de um modelo treinado com Scikit-learn.
2. Registro do modelo no Model Store do BentoML.
3. Implementação de um serviço de inferência.
4. Construção de um Bento com o serviço e o modelo.
5. Containerização do Bento utilizando Docker.
6. Execução e teste da API de inferência.
7. Integração da API com uma interface Streamlit.

## 3. Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3.11 | Linguagem de programação |
| BentoML 1.4.40 | Gerenciamento, empacotamento e disponibilização do modelo |
| Scikit-learn | Treinamento e pré-processamento do modelo |
| Pandas e NumPy | Manipulação dos dados |
| Docker | Containerização e execução do serviço |
| Streamlit | Interface para demonstração da aplicação |
| Requests | Comunicação HTTP entre a interface e a API |

## 4. Aplicação demonstrativa: Customer Churn Prediction

Para demonstrar o funcionamento do BentoML, foi desenvolvido um modelo de previsão de cancelamento de clientes de uma empresa de telecomunicações.

O modelo utiliza o conjunto de dados IBM Telco Customer Churn, composto por 7.043 registros e 21 colunas, tendo `Churn` como variável-alvo.

**Modelo utilizado:** Random Forest Classifier, desenvolvido com Scikit-learn.

O pipeline inclui:
- Imputação de valores ausentes.
- Codificação de variáveis categóricas com One-Hot Encoding.
- Pré-processamento com `ColumnTransformer` e `Pipeline`.
- Treinamento e avaliação do classificador.

O modelo foi avaliado em um conjunto de teste correspondente a 20% dos dados. Entre os resultados obtidos, destacam-se Accuracy de 77,08% e ROC-AUC de 82,29%.

Essas informações contextualizam o modelo utilizado, mas não representam o foco principal do trabalho.

## 5. Arquitetura da demonstração

O fluxo de funcionamento pode ser representado da seguinte forma:

```text
Modelo treinado com Scikit-learn
              |
              v
      BentoML Model Store
              |
              v
       Serviço BentoML
              |
              v
         Bento Build
              |
              v
      Imagem Docker
              |
              v
        API de inferência
              |
              v
       Interface Streamlit
```

O treinamento do modelo é realizado separadamente com Scikit-learn. O BentoML entra posteriormente, permitindo registrar o modelo, incorporá-lo a um serviço e empacotar os componentes necessários para sua execução.

## 6. Estrutura do projeto

```text
bentoml-churn/
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── models/
│   └── churn_pipeline.joblib
├── reports/
│   ├── churn_by_contract.png
│   ├── churn_distribution.png
│   └── tenure_distribution.png
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── service.py
│   └── train.py
├── .gitignore
├── bentofile.yaml
├── README.md
└── requirements.txt
```

O arquivo `models/churn_pipeline.joblib` é gerado durante o treinamento e não precisa ser versionado no Git. O modelo também é registrado no Model Store do BentoML.

## 7. Ambiente de execução

Pré-requisitos:

- Python 3.11.
- Conda ou ambiente virtual Python.
- Docker instalado e em execução.
- Git.

Clone o repositório:

```bash
git clone https://github.com/nathaliavsmarques/bentoml-na-pratica.git
cd bentoml-na-pratica
```

Crie e ative o ambiente Conda:

```bash
conda create -n bentoml-churn python=3.11 -y
conda activate bentoml-churn
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 8. Preparação do modelo demonstrativo

Esta etapa utiliza Scikit-learn para preparar e treinar o modelo que será posteriormente gerenciado pelo BentoML.

### 8.1. Análise exploratória

```bash
python src/eda.py
```

Os gráficos gerados são armazenados em `reports/`.

### 8.2. Treinamento

```bash
python src/train.py
```

O pipeline treinado será salvo em:

```text
models/churn_pipeline.joblib
```

**Observação:** o treinamento é uma etapa preparatória da demonstração. O BentoML não é responsável por treinar o Random Forest neste projeto.

## 9. Demonstração do BentoML

Esta é a etapa central do trabalho.

### 9.1. Registro do modelo no Model Store

O primeiro passo é registrar o pipeline treinado no armazenamento de modelos do BentoML.

Execute:

```bash
python -c "import joblib, bentoml; model=joblib.load('models/churn_pipeline.joblib'); saved=bentoml.sklearn.save_model('churn_pipeline', model); print(saved.tag)"
```

O comando carrega o pipeline salvo pelo Scikit-learn e registra o modelo no Model Store.

O BentoML gera uma identificação específica para o modelo, semelhante a:

```text
churn_pipeline:xxxxxxxxxxxxxxxx
```

Para consultar os modelos registrados:

```bash
bentoml models list
```

**Importante:** a identificação gerada pode variar entre ambientes e registros. Utilize a tag exibida no seu terminal nas próximas configurações.

### 9.2. Implementação do serviço de inferência

O serviço está implementado em `src/service.py`.

Nele, o BentoML utiliza o recurso `BentoModel` para carregar o modelo registrado no Model Store.

A classe `ChurnService` define o serviço e disponibiliza o método `predict`, responsável por:

- Receber os dados do cliente.
- Verificar a presença dos campos obrigatórios.
- Preparar os dados para inferência.
- Executar a previsão com o modelo.
- Retornar a classificação e a probabilidade estimada de cancelamento.

A anotação `@bentoml.service` declara o serviço, enquanto `@bentoml.api` identifica o método disponibilizado como endpoint da API.

Antes de continuar, atualize a variável `MODEL_TAG` em `src/service.py` com a tag gerada no registro do modelo.

### 9.3. Configuração do Bento

O arquivo `bentofile.yaml` contém as configurações utilizadas pelo BentoML para construir o artefato.

Neste projeto, ele define:

- O serviço que será empacotado.
- Os arquivos do projeto que devem ser incluídos.
- A versão do modelo registrada no Model Store.
- As dependências Python necessárias para execução.

Atualize também a referência do modelo em `bentofile.yaml`, utilizando exatamente a mesma tag configurada em `src/service.py`.

### 9.4. Construção do Bento

Com o serviço e as referências configuradas, execute:

```bash
bentoml build
```

O BentoML empacota o serviço, o modelo e os componentes necessários para sua execução.

Para consultar os Bentos disponíveis:

```bash
bentoml list
```

Cada construção gera uma identificação própria, que será utilizada na etapa de containerização.

### 9.5. Containerização com Docker

O BentoML permite gerar uma imagem Docker a partir de um Bento.

Utilize a identificação gerada na etapa anterior:

```bash
bentoml containerize NOME_DO_BENTO:VERSAO
```

Exemplo:

```bash
bentoml containerize churn_prediction:VERSAO
```

O comando utiliza o Docker para construir uma imagem que contém o serviço e o modelo empacotado.

Para consultar as imagens disponíveis:

```bash
docker images
```

### 9.6. Execução do serviço em container

Execute a imagem gerada:

```bash
docker run -d \
  --name churn-prediction-api \
  -p 3000:3000 \
  churn_prediction:VERSAO
```

Verifique se o container está em execução:

```bash
docker ps
```

Consulte os logs:

```bash
docker logs churn-prediction-api
```

A API estará disponível em:

`http://localhost:3000`

A documentação OpenAPI em JSON estará disponível em:

`http://localhost:3000/docs.json`

Para interromper e remover o container:

```bash
docker stop churn-prediction-api
docker rm churn-prediction-api
```

## 10. Teste da API

A API disponibiliza o endpoint `POST /predict`.

Exemplo de requisição:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 2,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 70.70,
      "TotalCharges": 141.40
    }
  }'
```

A resposta apresenta a classificação, a probabilidade estimada de cancelamento e uma mensagem explicativa.

## 11. Integração com Streamlit

Como complemento à demonstração da API, foi desenvolvida uma interface com Streamlit.

Com o container da API em execução, abra outro terminal, ative o ambiente Conda e execute:

```bash
conda activate bentoml-churn
streamlit run app/streamlit_app.py
```

A interface estará disponível em:

`http://localhost:8501`

O Streamlit envia os dados preenchidos pelo usuário para o endpoint BentoML e apresenta o resultado retornado pela API.

Essa integração demonstra como um serviço disponibilizado pelo BentoML pode ser consumido por outra aplicação.

## 12. O que foi demonstrado com o BentoML

Ao longo do projeto, foram explorados os seguintes recursos:

| Recurso | Aplicação no projeto |
|---|---|
| Model Store | Registro do pipeline Scikit-learn |
| BentoModel | Carregamento do modelo registrado |
| Service | Definição do serviço de inferência |
| API | Disponibilização do método de previsão |
| bentofile.yaml | Configuração dos componentes do Bento |
| Bento Build | Empacotamento do serviço e do modelo |
| Containerize | Geração da imagem Docker |
| Docker | Execução do serviço em container |

## 13. Considerações finais

A demonstração evidencia como o BentoML pode ser utilizado para transformar um modelo de Machine Learning previamente treinado em um serviço de inferência organizado e empacotado.

O projeto Customer Churn Prediction foi escolhido como exemplo para tornar esse processo concreto, permitindo acompanhar o registro do modelo, a implementação do serviço, a construção do Bento e sua execução em container Docker.

Esses recursos podem ser aplicados a diferentes projetos de Machine Learning, independentemente do problema de negócio, desde que o modelo e suas dependências sejam compatíveis com o ambiente de execução.

O trabalho possui finalidade acadêmica e demonstrativa, não representando uma implantação em ambiente de produção.
