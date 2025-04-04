import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Configuração da interface
st.set_page_config(page_title="Análise Bayesiana de Teste A/B", layout="centered")
st.title("📊 Análise Bayesiana de Teste A/B")
st.write("Este aplicativo calcula a probabilidade de uma variação ser melhor que o controle com um modelo Bayesiano.")

# Layout em colunas para separar os grupos
col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Grupo Controle")
    sessions_control = st.number_input("Sessões (Controle)", min_value=1, value=14403)
    revenue_control = st.number_input("Receita Total (Controle)", min_value=0.01, value=803801.41)
    rpv_control_manual = st.number_input("RPV (Receita por Visita - Controle)", min_value=0.01, value=revenue_control / sessions_control)

with col2:
    st.header("🚀 Grupo Variação")
    sessions_variation = st.number_input("Sessões (Variação)", min_value=1, value=13913)
    revenue_variation = st.number_input("Receita Total (Variação)", min_value=0.01, value=838703.13)
    rpv_variation_manual = st.number_input("RPV (Receita por Visita - Variação)", min_value=0.01, value=revenue_variation / sessions_variation)

# Determinar qual RPV usar (manual ou calculado)
rpv_control = rpv_control_manual if rpv_control_manual > 0 else revenue_control / sessions_control
rpv_variation = rpv_variation_manual if rpv_variation_manual > 0 else revenue_variation / sessions_variation

# Cálculo do desvio padrão estimado
std_control = rpv_control / np.sqrt(sessions_control)
std_variation = rpv_variation / np.sqrt(sessions_variation)

# Simulação Bayesiana
np.random.seed(42)
samples_control = np.random.normal(rpv_control, std_control, 100000)
samples_variation = np.random.normal(rpv_variation, std_variation, 100000)

# Probabilidade de a variação ser melhor
prob_variation_better = np.mean(samples_variation > samples_control)

# Explicação do resultado
if prob_variation_better > 0.95:
    explanation = "🎉 A variação tem uma **alta chance** de ser melhor que o controle! Você pode considerar implementá-la."
elif prob_variation_better > 0.75:
    explanation = "✅ A variação tem uma **boa chance** de ser melhor que o controle, mas pode valer a pena continuar testando."
elif prob_variation_better > 0.55:
    explanation = "⚖️ Os resultados são **inconclusivos**. Não há evidência forte o suficiente para tomar uma decisão."
else:
    explanation = "❌ O controle **ainda é a melhor opção**. A variação não demonstrou melhora significativa."

# Exibição dos resultados
st.divider()
st.subheader("📈 Resultados da Análise")
st.metric("RPV Controle", f"R$ {rpv_control:.2f}")
st.metric("RPV Variação", f"R$ {rpv_variation:.2f}")
st.metric("Probabilidade da Variação ser melhor", f"{prob_variation_better:.2%}")

# Exibir a explicação do resultado
st.subheader("📝 Explicação do Resultado")
st.info(explanation)

# Criar gráfico da distribuição
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(samples_control, bins=50, alpha=0.5, label="Controle", density=True)
ax.hist(samples_variation, bins=50, alpha=0.5, label="Variação", density=True)
ax.axvline(rpv_control, color='blue', linestyle='dashed', linewidth=2, label="Média Controle")
ax.axvline(rpv_variation, color='orange', linestyle='dashed', linewidth=2, label="Média Variação")
ax.legend()
ax.set_title("Distribuição Bayesiana do RPV")
ax.set_xlabel("RPV")
ax.set_ylabel("Densidade")

# Mostrar gráfico no Streamlit
st.pyplot(fig)
