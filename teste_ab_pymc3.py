import streamlit as st
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Configuração da interface
st.set_page_config(page_title="Análise Bayesiana de Teste A/B - PyMC3", layout="centered")
st.title("📊 Análise Bayesiana de Teste A/B (PyMC3)")
st.write("Este aplicativo usa inferência Bayesiana com PyMC3 para calcular a probabilidade de uma variação ser melhor que o controle.")

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Grupo Controle")
    sessions_control = st.number_input("Sessões (Controle)", min_value=1, value=14403)
    revenue_control = st.number_input("Receita Total (Controle)", min_value=0.01, value=803801.41)
    rpv_control = revenue_control / sessions_control

with col2:
    st.header("🚀 Grupo Variação")
    sessions_variation = st.number_input("Sessões (Variação)", min_value=1, value=13913)
    revenue_variation = st.number_input("Receita Total (Variação)", min_value=0.01, value=838703.13)
    rpv_variation = revenue_variation / sessions_variation

# Modelo Bayesiano
with pm.Model() as model:
    mu_control = pm.Normal("mu_control", mu=rpv_control, sigma=rpv_control / np.sqrt(sessions_control))
    mu_variation = pm.Normal("mu_variation", mu=rpv_variation, sigma=rpv_variation / np.sqrt(sessions_variation))
    diff = pm.Deterministic("diff", mu_variation - mu_control)
    trace = pm.sample(10000, return_inferencedata=False, progressbar=True)

# Cálculo da probabilidade
prob_variation_better = np.mean(trace["diff"] > 0)

# Explicação do resultado
if prob_variation_better > 0.95:
    explanation = "🎉 A variação tem uma **alta chance** de ser melhor que o controle! Você pode considerar implementá-la."
elif prob_variation_better > 0.75:
    explanation = "✅ A variação tem uma **boa chance** de ser melhor que o controle, mas pode valer a pena continuar testando."
elif prob_variation_better > 0.55:
    explanation = "⚖️ Os resultados são **inconclusivos**. Não há evidência forte o suficiente para tomar uma decisão."
else:
    explanation = "❌ O controle **ainda é a melhor opção**. A variação não demonstrou melhora significativa."

# Exibir resultados
st.divider()
st.subheader("📈 Resultados da Análise")
st.metric("RPV Controle", f"R$ {rpv_control:.2f}")
st.metric("RPV Variação", f"R$ {rpv_variation:.2f}")
st.metric("Probabilidade da Variação ser melhor", f"{prob_variation_better:.2%}")

st.subheader("📝 Explicação do Resultado")
st.info(explanation)

# Gráfico da distribuição
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(trace["diff"], bins=50, alpha=0.7, color="blue")
ax.axvline(0, color='red', linestyle='dashed', linewidth=2, label="Nenhuma Diferença")
ax.set_title("Distribuição Posterior da Diferença de RPV")
ax.set_xlabel("Diferença de RPV")
ax.set_ylabel("Frequência")
ax.legend()
st.pyplot(fig)

# Explicação detalhada
with st.expander("📊 Como os cálculos foram realizados?"):
    st.markdown("""
    ### 1️⃣ Cálculo da Receita por Visita (RPV)
    
    O RPV é calculado dividindo a **Receita Total** pelo número de **Sessões**:
    \[
    RPV = \frac{\text{Receita Total}}{\text{Sessões}}
    \]
    
    ### 2️⃣ Modelagem Bayesiana com PyMC3
    
    Assumimos que o RPV segue uma distribuição Normal com média desconhecida. Modelamos:
    
    \[
    \mu_{control} \sim \mathcal{N}(RPV_{control}, \frac{RPV_{control}}{\sqrt{Sessões}})
    \]
    \[
    \mu_{variação} \sim \mathcal{N}(RPV_{variação}, \frac{RPV_{variação}}{\sqrt{Sessões}})
    \]
    
    ### 3️⃣ Inferência com Amostragem MCMC
    
    Geramos 10.000 amostras para estimar a distribuição da diferença entre os RPV:
    
    \[
    \text{Diff} = \mu_{variação} - \mu_{control}
    \]
    
    ### 4️⃣ Cálculo da Probabilidade Bayesiana
    
    A fração de amostras onde a variação supera o controle nos dá a probabilidade desejada:
    
    \[
    P(\mu_{variação} > \mu_{control}) = \frac{\sum (\mu_{variação} > \mu_{control})}{10.000}
    \]
    """)