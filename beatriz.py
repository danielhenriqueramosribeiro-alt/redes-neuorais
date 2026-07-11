import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuração da página da aplicação (Tema Personalizado da Bia)
st.set_page_config(
    page_title="O Oráculo de Inteligência Artificial da Beatriz",
    page_icon="👑",
    layout="centered"
)

st.title("👑 BiaOsfera IA - O Oráculo da Beatriz")
st.markdown("Analisando cientificamente a rotina, os surtos e os dramas da **Bia** através de Machine Learning! 🤖✨")
st.write("---")

# Menu de navegação lateral com a versão adaptada para a Beatriz (Agora com 11 opções)
atividade = st.sidebar.selectbox(
    "Escolha o Diagnóstico da Bia:",
    [
        "1. IA das Notas vs Estudo da Bia",
        "2. Detector de Sono da Bia Fofocando",
        "3. IA do Humor da Bia e o Calor",
        "4. Detector de Ranço da Bia (Aprovação)",
        "5. IA da Felicidade do Pet da Bia",
        "6. Detector de Séries/Filmes da Bia",
        "7. IA dos Vestidos da Bia (Atraso Estimado)",
        "8. IA do Tamanho da Compra Impulsiva",
        "9. Detector de Música de Bad da Bia",
        "10. IA da Energia Baseada em Cafés",
        "11. Rede Neural: O Nível de Surtos da Bia"
    ]
)

# -----------------------------------------------------------------
if atividade == "1. IA das Notas vs Estudo da Bia":
    st.subheader("📝 1. IA das Notas vs Estudo da Bia")
    st.markdown("**Objetivo Original:** Prever nota baseada nas horas de estudo.  \n**Versão Bia:** Quantas horas a Bia realmente estudou vs a nota que ela vai tirar fingindo que sabe a matéria.")
    
    estudos = pd.DataFrame({'notas': [1,2,4,6,8,10], 'horas': [2,4,5,7,9,10]})
    modelo = LinearRegression().fit(estudos[['horas']], estudos['notas'])
    
    horas_input = st.number_input("Quantas horas a Bia passou olhando pro livro enquanto mexia no celular?", min_value=0.0, max_value=24.0, value=6.0)
    if st.button("Prever Nota da gata"):
        predicao = modelo.predict([[horas_input]])[0]
        st.success(f"Nota prevista: {predicao:.2f}. Se for acima de 6, foi na base do milagre ou do chute certeiro!")

# -----------------------------------------------------------------
elif atividade == "2. Detector de Sono da Bia Fofocando":
    st.subheader("🎮 2. Detector de Sono da Bia Fofocando")
    st.markdown("**Objetivo Original:** Prever nível de cansaço baseado em horas jogando.  \n**Versão Bia:** Prever o nível de exaustão da Bia baseado em quantas horas ela passou fofocando no WhatsApp.")
    
    gamer = pd.DataFrame({'horas_jogo': [1,2,4,6,8,10], 'cansaco': [1,2,3,5,8,10]})
    modelo = LinearRegression().fit(gamer[['horas_jogo']], gamer['cansaco'])
    
    horas_input = st.number_input("Quantas horas de fofoca acumulada com as amigas hoje?", min_value=0.0, max_value=24.0, value=5.0)
    if st.button("Calcular Cansaço da Bia"):
        predicao = modelo.predict([[horas_input]])[0]
        st.warning(f"Nível de cansaço mental: {predicao:.2f} / 10.0. Alerta: Ela vai mandar áudio de 5 minutos dizendo que tá exausta.")

# -----------------------------------------------------------------
elif atividade == "3. IA do Humor da Bia e o Calor":
    st.subheader("🍦 3. IA do Humor da Bia e o Calor")
    st.markdown("**Objetivo Original:** Prever quantidade de sorvetes vendidos pela temperatura.  \n**Versão Bia:** Quantos sorvetes (ou açaís) a Bia precisa tomar para recuperar a paciência dependendo do calor.")
    
    sorvete = pd.DataFrame({'temperatura': [18,20,24,27,30,35], 'vendas': [20,25,40,55,70,100]})
    modelo = LinearRegression().fit(sorvete[['temperatura']], sorvete['vendas'])
    
    temp_input = st.number_input("Qual a temperatura atual lá fora? (°C):", min_value=-10.0, max_value=50.0, value=32.0)
    if st.button("Calcular Dose de Açúcar Emocional"):
        predicao = modelo.predict([[temp_input]])[0]
        st.info(f"A IA calculou: A Bia precisa de {predicao:.1f} gramas de açaí/sorvete para não responder ninguém com patada.")

# -----------------------------------------------------------------
elif atividade == "4. Detector de Ranço da Bia (Aprovação)":
    st.subheader("🥷 4. Detector de Ranço da Bia")
    st.markdown("**Objetivo Original:** Classificar em aprovado/reprovado usando Regressão Logística.  \n**Versão Bia:** Quantos vacilos (respostas secas) a pessoa deu com a Bia e se ela já pegou ranço definitivo.")
    
    alunos = pd.DataFrame({'faltas': [0,1,2,5,7,10], 'resultado': [1,1,1,0,0,0]})
    modelo = LogisticRegression().fit(alunos[['faltas']], alunos['resultado'])
    
    faltas_input = st.number_input("Quantas vezes o indivíduo visualizou e não respondeu a Bia?", min_value=0, max_value=50, value=3)
    if st.button("Analisar Status de Ranço"):
        predicao = modelo.predict([[faltas_input]])[0]
        resultado = "🟢 SALVO (A Bia ainda tolera a existência da pessoa)" if predicao == 1 else "🔴 RANÇO INSTALADO (Já foi silenciado no Instagram e arquivado no WhatsApp)"
        st.subheader(f"Resultado: {resultado}")

# -----------------------------------------------------------------
elif atividade == "5. IA da Felicidade do Pet da Bia":
    st.subheader("🐶 5. IA da Felicidade do Pet da Bia")
    st.markdown("**Objetivo Original:** Prever felicidade do cachorro.  \n**Versão Bia:** Prever a felicidade do pet da Bia baseado em quantos mimos ela comprou para ele essa semana.")
    
    pets = pd.DataFrame({'passeios': [1,2,3,4,5], 'felicidade': [2,4,5,8,10]})
    modelo = LinearRegression().fit(pets[['passeios']], pets['felicidade'])
    
    passeios_input = st.number_input("Quantos sachês/mimos você deu pro bichinho essa semana, Bia?", min_value=0, max_value=20, value=3)
    if st.button("Analisar Humor do Pet"):
        predicao = modelo.predict([[passeios_input]])[0]
        st.success(f"Nível de felicidade do pet: {predicao:.2f} / 10.0. (Se for baixo, ele vai morder o carregador do seu celular).")

# -----------------------------------------------------------------
elif atividade == "6. Detector de Séries/Filmes da Bia":
    st.subheader("🎬 6. Detector de Séries/Filmes da Bia")
    st.markdown("**Objetivo Original:** Prever nota do filme usando duração.  \n**Versão Bia:** Prever a nota que a Bia vai dar para um filme baseado no tempo de duração (ela odeia filme muito longo).")
    
    filmes = pd.DataFrame({'duracao': [80,90,100,110,120], 'nota': [4,5,7,8,9]})
    modelo = LinearRegression().fit(filmes[['duracao']], filmes['nota'])
    
    duracao_input = st.number_input("Duração do filme/episódio recomendado para a Bia (minutos):", min_value=30, max_value=300, value=95)
    if st.button("Prever Nota da Bia"):
        predicao = modelo.predict([[duracao_input]])[0]
        st.info(f"Nota prevista da Bia: {predicao:.2f} / 10.0. Se passar de 2 horas, ela desiste nos primeiros 15 minutos.")

# -----------------------------------------------------------------
elif atividade == "7. IA dos Vestidos da Bia (Atraso Estimado)":
    st.subheader("👗 7. IA dos Vestidos da Bia")
    st.markdown("**Objetivo Original:** Prever preço/métrica linear.  \n**Versão Bia:** Prever quantos minutos a Bia vai atrasar para sair baseado em quantos vestidos ela tirou do armário, experimentou e jogou em cima da cama dizendo que 'não tem roupa'.")
    
    # Dados matemáticos adaptados: vestidos vs minutos de atraso acumulados
    dados_vestidos = pd.DataFrame({'vestidos': [1, 2, 3, 4, 5], 'minutos_atraso': [15, 30, 45, 60, 90]})
    modelo = LinearRegression().fit(dados_vestidos[['vestidos']], dados_vestidos['minutos_atraso'])
    
    vestidos_input = st.number_input("Quantos vestidos a Bia experimentou e rejeitou até agora?", min_value=1, max_value=20, value=3)
    if st.button("Calcular Hora do Rolê"):
        predicao = modelo.predict([[vestidos_input]])[0]
        st.error(f"Aviso de Atraso: A Bia vai demorar mais {predicao:.1f} minutos para ficar pronta. Pode esperar sentado!")

# -----------------------------------------------------------------
elif atividade == "8. IA do Tamanho da Compra Impulsiva":
    st.subheader("🛍️ 8. IA do Tamanho da Compra Impulsiva")
    st.markdown("**Objetivo Original:** Prever preço da pizza pelo tamanho.  \n**Versão Bia:** Prever o valor da fatura do cartão da Bia baseado no tamanho do 'surto consumista' dela na Shein/Shopee.")
    
    pizza = pd.DataFrame({'tamanho': [20,25,30,35,40], 'preco': [20,30,40,50,60]})
    modelo = LinearRegression().fit(pizza[['tamanho']], pizza['preco'])
    
    tamanho_input = st.number_input("Nível do surto de compras da Bia (em quantidade de itens adicionados ao carrinho):", min_value=10, max_value=80, value=30)
    if st.button("Prever Rombo na Fatura"):
        predicao = modelo.predict([[tamanho_input]])[0]
        st.success(f"Valor estimado do estrago: R$ {predicao:.2f}. Quem converte não diverte, né Bia?")

# -----------------------------------------------------------------
elif atividade == "9. Detector de Música de Bad da Bia":
    st.subheader("🎵 9. Detector de Música de Bad da Bia")
    st.markdown("**Objetivo Original:** Prever chance da música viralizar baseado no BPM.  \n**Versão Bia:** Medir o potencial dramático e o risco da Bia chorar no quarto ouvindo músicas lentas.")
    
    musica = pd.DataFrame({'bpm': [80,90,100,120,140], 'viral': [1,2,4,7,10]})
    modelo = LinearRegression().fit(musica[['bpm']], musica['viral'])
    
    bpm_input = st.number_input("Qual o BPM (ritmo) da música que a Bia botou no fone? (Músicas lentas = menor BPM)", min_value=40, max_value=250, value=85)
    if st.button("Calcular Índice de Drama"):
        predicao = modelo.predict([[bpm_input]])[0]
        st.warning(f"Risco de Story reflexivo no Instagram: {predicao:.2f} / 10.0. Tira a Taylor Swift dela imediatamente!")

# -----------------------------------------------------------------
elif atividade == "10. IA da Energia Baseada em Cafés":
    st.subheader("☕ 10. IA da Energia Baseada em Cafés")
    st.markdown("**Objetivo Original:** Prever energia baseada em cafés tomados.  \n**Versão Bia:** Prever o nível de animação e hiperatividade da Bia dependendo de quantas xícaras de café ela tomou.")
    
    cafe = pd.DataFrame({'xicaras': [1,2,3,4,5], 'energia': [2,4,6,8,10]})
    modelo = LinearRegression().fit(cafe[['xicaras']], cafe['energia'])
    
    xicaras_input = st.number_input("Quantas xícaras de café a Bia tomou hoje?", min_value=0, max_value=15, value=2)
    if st.button("Medir Energia da gata"):
        predicao = modelo.predict([[xicaras_input]])[0]
        st.success(f"Nível de energia da Bia: {predicao:.2f} / 10.0. Acima de 6 ela vira uma máquina de falar.")

# -----------------------------------------------------------------
elif atividade == "11. Rede Neural: O Nível de Surtos da Bia":
    st.subheader("🦹‍♂️ 11. Rede Neural: O Nível de Surtos da Bia")
    st.markdown("**Objetivo Original:** Classificar herói forte ou fraco usando Rede Neural MLP.  \n**Versão Bia:** Uma inteligência artificial com neurônios artificiais para prever se ela vai surtar ou manter a calma diante de um problema.")
    
    herois = pd.DataFrame({'forca': [1,2,3,7,8,10], 'heroi': [0,0,0,1,1,1]})
    modelo = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42)
    modelo.fit(herois[['forca']], herois['heroi'])
    
    forca_input = st.number_input("Nível de estresse do problema (1 a 10):", min_value=1, max_value=10, value=5)
    if st.button("Consultar Rede Neural de Emoções da Bia"):
        predicao = modelo.predict([[forca_input]])[0]
        status = "🚨 SURTO COMPLETO (Vai mandar áudio chorando e rindo ao mesmo tempo)" if predicao == 1 else "🧘‍♀️ BIA PLENA ('É sobre isso e tá tudo bem')"
        st.subheader(f"A Inteligência Artificial previu: {status}")