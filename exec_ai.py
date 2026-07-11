import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuração da página da aplicação
st.set_page_config(
    page_title="Vibecode - Exercícios IA",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Painel Interativo - Exercícios de IA")
st.markdown("Trabalho desenvolvido para a disciplina de IA — **Vibecode**.")
st.write("---")

# Menu de navegação lateral para escolher a atividade
atividade = st.sidebar.selectbox(
    "Escolha a Atividade:",
    [
        "1. IA das Notas Escolares",
        "2. Detector de Sono Gamer",
        "3. IA do Sorvete",
        "4. Detector de Aprovação Ninja",
        "5. IA do Pet Feliz",
        "6. Detector de Filme Bom",
        "7. IA da Pizza",
        "8. Detector de Música Viral",
        "9. IA da Energia do Café",
        "10. Rede Neural dos Super-Heróis"
    ]
)

# -----------------------------------------------------------------
if atividade == "1. IA das Notas Escolares":
    st.subheader("📝 1. IA das Notas Escolares")
    st.markdown("**Objetivo:** Prever nota baseada nas horas de estudo.")
    
    estudos = pd.DataFrame({'notas': [1,2,4,6,8,10], 'horas': [2,4,5,7,9,10]})
    st.write("Dados de Treinamento:", estudos)
    
    modelo = LinearRegression().fit(estudos[['horas']], estudos['notas'])
    
    horas_input = st.number_input("Digite as horas de estudo:", min_value=0.0, max_value=24.0, value=6.0)
    if st.button("Prever Nota"):
        predicao = modelo.predict([[horas_input]])[0]
        st.success(f"Nota prevista: {predicao:.2f}")

# -----------------------------------------------------------------
elif atividade == "2. Detector de Sono Gamer":
    st.subheader("🎮 2. Detector de Sono Gamer")
    st.markdown("**Objetivo:** Prever nível de cansaço baseado em horas jogando.")
    
    gamer = pd.DataFrame({'horas_jogo': [1,2,4,6,8,10], 'cansaco': [1,2,3,5,8,10]})
    st.write("Dados de Treinamento:", gamer)
    
    modelo = LinearRegression().fit(gamer[['horas_jogo']], gamer['cansaco'])
    
    horas_input = st.number_input("Horas jogando:", min_value=0.0, max_value=24.0, value=5.0)
    if st.button("Calcular Cansaço"):
        predicao = modelo.predict([[horas_input]])[0]
        st.warning(f"Nível de cansaço estimado: {predicao:.2f} / 10.0")

# -----------------------------------------------------------------
elif atividade == "3. IA do Sorvete":
    st.subheader("🍦 3. IA do Sorvete")
    st.markdown("**Objetivo:** Prever quantidade de sorvetes vendidos pela temperatura.")
    
    sorvete = pd.DataFrame({'temperatura': [18,20,24,27,30,35], 'vendas': [20,25,40,55,70,100]})
    st.write("Dados de Treinamento:", sorvete)
    
    modelo = LinearRegression().fit(sorvete[['temperatura']], sorvete['vendas'])
    
    temp_input = st.number_input("Temperatura do dia (°C):", min_value=-10.0, max_value=50.0, value=28.0)
    if st.button("Prever Vendas"):
        predicao = modelo.predict([[temp_input]])[0]
        st.info(f"Expectativa de vendas: {predicao:.2f} sorvetes")

# -----------------------------------------------------------------
elif atividade == "4. Detector de Aprovação Ninja":
    st.subheader("🥷 4. Detector de Aprovação Ninja")
    st.markdown("**Objetivo:** Classificar se o aluno ninja está aprovado ou reprovado.")
    
    alunos = pd.DataFrame({'faltas': [0,1,2,5,7,10], 'resultado': [1,1,1,0,0,0]})
    st.write("Dados de Treinamento (1 = Aprovado, 0 = Reprovado):", alunos)
    
    modelo = LogisticRegression().fit(alunos[['faltas']], alunos['resultado'])
    
    faltas_input = st.number_input("Quantidade de faltas:", min_value=0, max_value=50, value=3)
    if st.button("Verificar Status"):
        predicao = modelo.predict([[faltas_input]])[0]
        resultado = "🟢 APROVADO" if predicao == 1 else "🔴 REPROVADO"
        st.subheader(f"Resultado: {resultado}")

# -----------------------------------------------------------------
elif atividade == "5. IA do Pet Feliz":
    st.subheader("🐶 5. IA do Pet Feliz")
    st.markdown("**Objetivo:** Prever felicidade do cachorro baseado no número de passeios.")
    
    pets = pd.DataFrame({'passeios': [1,2,3,4,5], 'felicidade': [2,4,5,8,10]})
    st.write("Dados de Treinamento:", pets)
    
    modelo = LinearRegression().fit(pets[['passeios']], pets['felicidade'])
    
    passeios_input = st.number_input("Quantidade de passeios:", min_value=0, max_value=20, value=3)
    if st.button("Medir Felicidade"):
        predicao = modelo.predict([[passeios_input]])[0]
        st.success(f"Nível de felicidade do cãozinho: {predicao:.2f} / 10.0")

# -----------------------------------------------------------------
elif atividade == "6. Detector de Filme Bom":
    st.subheader("🎬 6. Detector de Filme Bom")
    st.markdown("**Objetivo:** Prever nota do filme usando a duração em minutos.")
    
    filmes = pd.DataFrame({'duracao': [80,90,100,110,120], 'nota': [4,5,7,8,9]})
    st.write("Dados de Treinamento:", filmes)
    
    modelo = LinearRegression().fit(filmes[['duracao']], filmes['nota'])
    
    duracao_input = st.number_input("Duração do filme (minutos):", min_value=30, max_value=300, value=105)
    if st.button("Avaliar Filme"):
        predicao = modelo.predict([[duracao_input]])[0]
        st.info(f"Nota estimada para o filme: {predicao:.2f} / 10.0")

# -----------------------------------------------------------------
elif atividade == "7. IA da Pizza":
    st.subheader("🍕 7. IA da Pizza")
    st.markdown("**Objetivo:** Prever preço da pizza pelo tamanho (centímetros).")
    
    pizza = pd.DataFrame({'tamanho': [20,25,30,35,40], 'preco': [20,30,40,50,60]})
    st.write("Dados de Treinamento:", pizza)
    
    modelo = LinearRegression().fit(pizza[['tamanho']], pizza['preco'])
    
    tamanho_input = st.number_input("Tamanho da pizza (cm):", min_value=10, max_value=80, value=32)
    if st.button("Calcular Preço"):
        predicao = modelo.predict([[tamanho_input]])[0]
        st.success(f"Preço estimado: R$ {predicao:.2f}")

# -----------------------------------------------------------------
elif atividade == "8. Detector de Música Viral":
    st.subheader("🎵 8. Detector de Música Viral")
    st.markdown("**Objetivo:** Prever a chance de engajamento/viralização baseado no BPM.")
    
    musica = pd.DataFrame({'bpm': [80,90,100,120,140], 'viral': [1,2,4,7,10]})
    st.write("Dados de Treinamento:", musica)
    
    modelo = LinearRegression().fit(musica[['bpm']], musica['viral'])
    
    bpm_input = st.number_input("BPM da Música:", min_value=40, max_value=250, value=110)
    if st.button("Analisar Potencial Viral"):
        predicao = modelo.predict([[bpm_input]])[0]
        st.warning(f"Pontuação de potencial viral: {predicao:.2f} / 10.0")

# -----------------------------------------------------------------
elif atividade == "9. IA da Energia do Café":
    st.subheader("☕ 9. IA da Energia do Café")
    st.markdown("**Objetivo:** Prever energia baseada em xícaras de café tomadas.")
    
    cafe = pd.DataFrame({'xicaras': [1,2,3,4,5], 'energia': [2,4,6,8,10]})
    st.write("Dados de Treinamento:", cafe)
    
    modelo = LinearRegression().fit(cafe[['xicaras']], cafe['energia'])
    
    xicaras_input = st.number_input("Xícaras de café consumidas:", min_value=0, max_value=15, value=3)
    if st.button("Calcular Energia"):
        predicao = modelo.predict([[xicaras_input]])[0]
        st.success(f"Nível de energia estimado: {predicao:.2f} / 10.0")

# -----------------------------------------------------------------
elif atividade == "10. Rede Neural dos Super-Heróis":
    st.subheader("🦸‍♂️ 10. Rede Neural dos Super-Heróis")
    st.markdown("**Objetivo:** Classificar se um herói é Forte (1) ou Fraco (0) através de uma Rede Neural (MLP).")
    
    herois = pd.DataFrame({'forca': [1,2,3,7,8,10], 'heroi': [0,0,0,1,1,1]})
    st.write("Dados de Treinamento:", herois)
    
    modelo = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42)
    modelo.fit(herois[['forca']], herois['heroi'])
    
    forca_input = st.number_input("Nível de força do Herói (1 a 10):", min_value=1, max_value=10, value=5)
    if st.button("Classificar Força do Herói"):
        predicao = modelo.predict([[forca_input]])[0]
        status = "💪 FORTE" if predicao == 1 else "🍃 FRACO"
        st.subheader(f"A Rede Neural classificou como: {status}")