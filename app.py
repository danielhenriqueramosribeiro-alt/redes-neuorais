import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Configuração da página da aplicação (Tema Love/Meme)
st.set_page_config(
    page_title="Oráculo dos Namorados - IA",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Oráculo dos Namorados Meme - IA")
st.markdown("Quem precisa de terapia de casal quando se tem **Machine Learning**? 🤖")
st.write("---")

# Menu de navegação lateral com a versão adaptada dos 10 exercícios
atividade = st.sidebar.selectbox(
    "Escolha o Teste da IA:",
    [
        "1. IA do Romantismo vs Estudos",
        "2. Detector de Sono do Mozão Gamer",
        "3. IA do Humor e da Temperatura",
        "4. Detector de Aprovação da Sogra",
        "5. IA da Felicidade do Pet do Casal",
        "6. Detector de Filme que o Mozão Dorme",
        "7. IA do Tamanho da Pizza para Não Brigar",
        "8. Detector de Música que Vira Indireta",
        "9. IA da Energia Pós-Café do Casal",
        "10. Rede Neural: Quem manda na relação?"
    ]
)

# -----------------------------------------------------------------
if atividade == "1. IA do Romantismo vs Estudos":
    st.subheader("📝 1. IA do Romantismo vs Estudos")
    st.markdown("**Objetivo Original:** Prever nota baseada nas horas de estudo.  \n**Meme:** Quanto maior a nota na prova, mais tempo o mozão tem livre para você.")
    
    estudos = pd.DataFrame({'notas': [1,2,4,6,8,10], 'horas': [2,4,5,7,9,10]})
    modelo = LinearRegression().fit(estudos[['horas']], estudos['notas'])
    
    horas_input = st.number_input("Quantas horas o mozão fingiu que estudou?", min_value=0.0, max_value=24.0, value=6.0)
    if st.button("Prever Nota do Indivíduo"):
        predicao = modelo.predict([[horas_input]])[0]
        st.success(f"Nota prevista: {predicao:.2f}. Se for menor que 6, cancela o encontro de sábado para ele recuperar a média.")

# -----------------------------------------------------------------
elif atividade == "2. Detector de Sono do Mozão Gamer":
    st.subheader("🎮 2. Detector de Sono do Mozão Gamer")
    st.markdown("**Objetivo Original:** Prever nível de cansaço baseado em horas jogando.  \n**Meme:** Descubra se ele(a) vai dormir no meio da ligação hoje.")
    
    gamer = pd.DataFrame({'horas_jogo': [1,2,4,6,8,10], 'cansaco': [1,2,3,5,8,10]})
    modelo = LinearRegression().fit(gamer[['horas_jogo']], gamer['cansaco'])
    
    horas_input = st.number_input("Quantas horas o ser vivo passou no videogame hoje?", min_value=0.0, max_value=24.0, value=5.0)
    if st.button("Calcular Chance de Dormir na Call"):
        predicao = modelo.predict([[horas_input]])[0]
        st.warning(f"Nível de cansaço: {predicao:.2f} / 10.0. Acima de 7, o 'tô só descansando o olho' é uma mentira.")

# -----------------------------------------------------------------
elif atividade == "3. IA do Humor e da Temperatura":
    st.subheader("🍦 3. IA do Humor e da Temperatura")
    st.markdown("**Objetivo Original:** Prever quantidade de sorvetes vendidos pela temperatura.  \n**Meme:** Quantos sorvetes você precisa comprar para acalmar o mozão no calor de janeiro?")
    
    sorvete = pd.DataFrame({'temperatura': [18,20,24,27,30,35], 'vendas': [20,25,40,55,70,100]})
    modelo = LinearRegression().fit(sorvete[['temperatura']], sorvete['vendas'])
    
    temp_input = st.number_input("Qual a temperatura lá fora? (°C):", min_value=-10.0, max_value=50.0, value=32.0)
    if st.button("Calcular Custo da Paz"):
        predicao = modelo.predict([[temp_input]])[0]
        st.info(f"A IA sugere: compre exatamente {predicao:.1f} sorvetes para evitar discussões sobre onde ir jantar.")

# -----------------------------------------------------------------
elif atividade == "4. Detector de Aprovação da Sogra":
    st.subheader("🥷 4. Detector de Aprovação da Sogra")
    st.markdown("**Objetivo Original:** Classificar em aprovado ou reprovado usando regressão logística.  \n**Meme:** Quantos almoços de domingo você faltou na casa da sogra e o resultado do seu score com ela.")
    
    alunos = pd.DataFrame({'faltas': [0,1,2,5,7,10], 'resultado': [1,1,1,0,0,0]})
    modelo = LogisticRegression().fit(alunos[['faltas']], alunos['resultado'])
    
    faltas_input = st.number_input("Quantas vezes você inventou uma desculpa para não ir ver a família do mozão?", min_value=0, max_value=50, value=3)
    if st.button("Consultar Clima de Domingo"):
        predicao = modelo.predict([[faltas_input]])[0]
        resultado = "🟢 APROVADO (Ainda ganha tupperware com bolo)" if predicao == 1 else "🔴 REPROVADO (Falam mal de você no grupo da família sem você estar)"
        st.subheader(f"Status: {resultado}")

# -----------------------------------------------------------------
elif atividade == "5. IA da Felicidade do Pet do Casal":
    st.subheader("🐶 5. IA da Felicidade do Pet do Casal")
    st.markdown("**Objetivo Original:** Prever felicidade do cachorro.  \n**Meme:** Se o cachorro que vocês adotaram juntos está feliz ou planeja fugir de casa.")
    
    pets = pd.DataFrame({'passeios': [1,2,3,4,5], 'felicidade': [2,4,5,8,10]})
    modelo = LinearRegression().fit(pets[['passeios']], pets['felicidade'])
    
    passeios_input = st.number_input("Quantas vezes vocês levaram o pet para passear essa semana?", min_value=0, max_value=20, value=3)
    if st.button("Analisar Humor do Dog"):
        predicao = modelo.predict([[passeios_input]])[0]
        st.success(f"Nível de felicidade do doguinho: {predicao:.2f} / 10.0. Nota: Se o índice for baixo, ele vai roer o chinelo preferido do namorado.")

# -----------------------------------------------------------------
elif atividade == "6. Detector de Filme que o Mozão Dorme":
    st.subheader("🎬 6. Detector de Filme que o Mozão Dorme")
    st.markdown("**Objetivo Original:** Prever nota do filme usando duração.  \n**Meme:** Quanto maior a duração do filme que você escolhe, menor a nota que o mozão dá pra sua escolha porque ele dorme no meio.")
    
    filmes = pd.DataFrame({'duracao': [80,90,100,110,120], 'nota': [4,5,7,8,9]})
    modelo = LinearRegression().fit(filmes[['duracao']], filmes['nota'])
    
    duracao_input = st.number_input("Duração do filme que você quer assistir (minutos):", min_value=30, max_value=300, value=120)
    if st.button("Prever Nota do Cônjuge"):
        predicao = modelo.predict([[duracao_input]])[0]
        st.info(f"Nota prevista pelo mozão: {predicao:.2f} / 10.0 (Geralmente cai drasticamente se passar de 90 minutos de tédio).")

# -----------------------------------------------------------------
elif atividade == "7. IA do Tamanho da Pizza para Não Brigar":
    st.subheader("🍕 7. IA do Tamanho da Pizza para Não Brigar")
    st.markdown("**Objetivo Original:** Prever preço da pizza pelo tamanho.  \n**Meme:** Previsão do rombo na carteira dependendo do tamanho da fome de vocês dois juntos.")
    
    pizza = pd.DataFrame({'tamanho': [20,25,30,35,40], 'preco': [20,30,40,50,60]})
    modelo = LinearRegression().fit(pizza[['tamanho']], pizza['preco'])
    
    tamanho_input = st.number_input("Qual o tamanho da pizza para alimentar esse amor? (cm):", min_value=10, max_value=80, value=35)
    if st.button("Verificar a Conta"):
        predicao = modelo.predict([[tamanho_input]])[0]
        st.success(f"Preço do jantar: R$ {predicao:.2f}. Ideal dividir no pix para não pesar o orçamento do namoro.")

# -----------------------------------------------------------------
elif atividade == "8. Detector de Música que Vira Indireta":
    st.subheader("🎵 8. Detector de Música que Vira Indireta")
    st.markdown("**Objetivo Original:** Prever chance da música viralizar baseado no BPM.  \n**Meme:** Medidor de perigo de uma música rápida virar storie com indireta no Instagram.")
    
    musica = pd.DataFrame({'bpm': [80,90,100,120,140], 'viral': [1,2,4,7,10]})
    modelo = LinearRegression().fit(musica[['bpm']], musica['viral'])
    
    bpm_input = st.number_input("Qual o ritmo (BPM) da música triste que o mozão está ouvindo?", min_value=40, max_value=250, value=90)
    if st.button("Calcular Potencial de Briga por Storie"):
        predicao = modelo.predict([[bpm_input]])[0]
        st.warning(f"Pontuação de risco de drama: {predicao:.2f} / 10.0. Músicas lentas têm maior probabilidade de DR.")

# -----------------------------------------------------------------
elif atividade == "9. IA da Energia Pós-Café do Casal":
    st.subheader("☕ 9. IA da Energia Pós-Café do Casal")
    st.markdown("**Objetivo Original:** Prever energia baseada em cafés tomados.  \n**Meme:** Medidor de paciência para aguentar o mozão falando sem parar.")
    
    cafe = pd.DataFrame({'xicaras': [1,2,3,4,5], 'energia': [2,4,6,8,10]})
    modelo = LinearRegression().fit(cafe[['xicaras']], cafe['energia'])
    
    xicaras_input = st.number_input("Quantas xícaras de café o parceiro(a) tomou hoje?", min_value=0, max_value=15, value=2)
    if st.button("Verificar Nível de Hiperatividade"):
        predicao = modelo.predict([[xicaras_input]])[0]
        st.success(f"Nível de energia: {predicao:.2f} / 10.0. Prepare-se para ouvir fofocas do trabalho na velocidade 2x.")

# -----------------------------------------------------------------
elif atividade == "10. Rede Neural: Quem manda na relação?":
    st.subheader("🦹‍♂️ 10. Rede Neural: Quem manda na relação?")
    st.markdown("**Objetivo Original:** Classificar herói forte ou fraco usando Rede Neural MLP.  \n**Meme:** Classificar cientificamente quem ganha a discussão baseado no 'Nível de Argumentação' (Força).")
    
    herois = pd.DataFrame({'forca': [1,2,3,7,8,10], 'heroi': [0,0,0,1,1,1]})
    modelo = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=42)
    modelo.fit(herois[['forca']], herois['heroi'])
    
    forca_input = st.number_input("Seu nível de paciência/argumento na última DR (1 a 10):", min_value=1, max_value=10, value=4)
    if st.button("Consultar Veredito dos Neurônios Artificiais"):
        predicao = modelo.predict([[forca_input]])[0]
        status = "💪 VOCÊ MANDA (Desta vez a janta é onde você escolheu)" if predicao == 1 else "🍃 VOCÊ SÓ CONCORDA ('Sim, amor, você tá certo(a)')"
        st.subheader(f"A Rede Neural decidiu: {status}")