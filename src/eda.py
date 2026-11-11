
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Caminhos
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

# Carregamento
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ANÁLISE EXPLORATÓRIA - CUSTOMER CHURN")
print("=" * 60)

# 1. Dimensões
print("\n1. DIMENSÕES DO DATASET")
print(df.shape)

# 2. Tipos de dados
print("\n2. TIPOS DE DADOS")
print(df.dtypes)

# 3. Valores ausentes
print("\n3. VALORES AUSENTES")

# Tratar espaços vazios de TotalCharges para a análise
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print(df.isnull().sum())

# 4. Duplicatas
print("\n4. REGISTROS DUPLICADOS")
print("Linhas duplicadas:", df.duplicated().sum())
print("IDs duplicados:", df["customerID"].duplicated().sum())

# 5. Estatísticas numéricas
print("\n5. ESTATÍSTICAS DESCRITIVAS")
print(df.describe())

# 6. Variáveis categóricas
print("\n6. VARIÁVEIS CATEGÓRICAS")

categorical_columns = df.select_dtypes(
    include=["str", "object"]
).columns.tolist()


categorical_columns.remove("customerID")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

# 7. Cancelamento por tipo de contrato
print("\n7. CANCELAMENTO POR TIPO DE CONTRATO")

churn_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(churn_contract.round(2))

# 8. Tempo de permanência por cancelamento
print("\n8. TEMPO DE PERMANÊNCIA")

print(
    df.groupby("Churn")["tenure"]
    .describe()
    .round(2)
)

# 9. Valor mensal por cancelamento
print("\n9. VALOR MENSAL")

print(
    df.groupby("Churn")["MonthlyCharges"]
    .describe()
    .round(2)
)

# 10. Gráfico de distribuição do Churn
churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(7, 5))
churn_counts.plot(kind="bar")
plt.title("Distribuição de Cancelamento")
plt.xlabel("Cancelamento")
plt.ylabel("Quantidade de clientes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORTS_PATH / "churn_distribution.png")
plt.close()

# 11. Cancelamento por contrato
churn_contract.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Proporção de Churn por Tipo de Contrato")
plt.xlabel("Tipo de contrato")
plt.ylabel("Proporção (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORTS_PATH / "churn_by_contract.png")
plt.close()

# 12. Distribuição do tempo de permanência
plt.figure(figsize=(8, 5))

for churn in ["No", "Yes"]:
    df.loc[df["Churn"] == churn, "tenure"].plot(
        kind="hist",
        bins=20,
        alpha=0.5,
        label=churn
    )

plt.title("Tempo de Permanência por Churn")
plt.xlabel("Meses")
plt.ylabel("Frequência")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_PATH / "tenure_distribution.png")
plt.close()

print("\nAnálise concluída!")
print(f"Gráficos salvos em: {REPORTS_PATH}")
