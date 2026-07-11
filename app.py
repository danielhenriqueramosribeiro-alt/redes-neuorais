import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os

# Configuração da página do Streamlit (Design Corporativo)
st.set_page_config(
    page_title="Executive Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. CRIAÇÃO AUTOMÁTICA DA BASE DE DADOS (Simulação de Auditoria)
def garantir_base_dados():
    """Garante a existência do arquivo CSV com dados de serviços e reclamações."""
    if not os.path.exists('servicos_reclamacoes.csv'):
        np.random.seed(42)
        servicos = ['Suporte Técnico', 'Atendimento ao Cliente', 'Instalação de Equipamentos', 
                    'Manutenção Preventiva', 'Faturamento e Cobrança']
        
        dados = []
        for i in range(1, 151):  # 150 registros corporativos
            srv = np.random.choice(servicos)
            if srv in ['Suporte Técnico', 'Atendimento ao Cliente']:
                reclamacoes = np.random.randint(35, 75)
                tempo_res = np.random.uniform(24, 72)
                nota = np.random.uniform(1.0, 2.3)
            elif srv == 'Instalação de Equipamentos':
                reclamacoes = np.random.randint(15, 38)
                tempo_res = np.random.uniform(12, 36)
                nota = np.random.uniform(2.4, 3.7)
            else:
                reclamacoes = np.random.randint(0, 18)
                tempo_res = np.random.uniform(1, 10)
                nota = np.random.uniform(3.8, 5.0)
                
            dados.append([i, srv, reclamacoes, tempo_res, round(nota, 1)])
            
        df = pd.DataFrame(dados, columns=['id_servico', 'nome_servico', 'reclamacoes_mensais', 'tempo_resolucao_horas', 'nota_satisfacao'])
        df.to_csv('servicos_reclamacoes.csv', index=False)

# Garantir que a base exista antes de renderizar o app
garantir_base_dados()

# Cabeçalho da aplicação
st.title("🧠 Mapeamento de Qualidade Operacional com Redes Neurais")
st.markdown("""
O sistema analisa os históricos de fricção, tempos de resposta e feedbacks dos clientes para identificar 
padrões invisíveis através de Inteligência Artificial, gerando o diagnóstico preciso de cada setor da empresa.
""")

st.write("---")

# Seção de Ativação (Requisito: Botão para ativar a análise)
st.subheader("📊 Painel de Controle Operacional")
ativar_analise = st.button("Ativar Função de Análise", type="primary", use_container_width=True)

# CORREÇÃO DEFINITIVA: 'ativar_analise' sem a letra 'c'
if colocar_aqui := ativar_analise:  
    with st.spinner("Treinando os neurônios artificiais e processando dados..."):
        try:
            # Leitura via Pandas
            df = pd.read_csv('servicos_reclamacoes.csv')
            
            # Rotulagem lógica de negócio para o treinamento supervisionado
            def classificar_regras_negocio(row):
                if row['nota_satisfacao'] < 2.2 and row['reclamacoes_mensais'] > 45:
                    return 0  # serviço muito ruim
                elif row['nota_satisfacao'] < 3.6 or row['reclamacoes_mensais'] > 25:
                    return 1  # serviço ruim
                elif row['nota_satisfacao'] < 4.5:
                    return 2  # serviço bom
                else:
                    return 3  # serviço excelente

            y = df.apply(classificar_regras_negocio, axis=1)
            X = df[['reclamacoes_mensais', 'tempo_resolucao_horas', 'nota_satisfacao']]
            
            # Escalonamento/Normalização obrigatória
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Arquitetura de Rede Neural MLP (Scikit-Learn)
            mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
            mlp.fit(X_scaled, y)
            
            # Predição da IA
            df['classe_predita'] = mlp.predict(X_scaled)
            
            # Formato de saída conforme exigência pública
            mapeamento_niveis = {
                0: "serviço muito ruim",
                1: "serviço ruim",
                2: "serviço bom",
                3: "serviço excelente"
            }
            df['status_formatado'] = df['classe_predita'].map(mapeamento_niveis)
            
            # Consolidação executiva por setor
            resumo_executivo = df.groupby('nome_servico').agg({
                'reclamacoes_mensais': 'mean',
                'nota_satisfacao': 'mean',
                'status_formatado': lambda x: x.mode()[0]
            }).reset_index()
            
            # Exibição de Resultados na Janela Web do Streamlit
            st.success("Análise concluída com sucesso!")
            
            st.write("### 📋 Resultados Gerais por Setor")
            
            # Exibição organizada em cards em colunas dinâmicas
            colunas = st.columns(len(resumo_executivo))
            setores_criticos = []
            
            for index, row in resumo_executivo.iterrows():
                with colunas[index]:
                    status = row['status_formatado']
                    nome = row['nome_servico']
                    
                    # Definição de cores baseada na criticidade
                    if status == "serviço muito ruim":
                        st.error(f"**{nome}**\n\n🚨 {status.upper()}")
                        setores_criticos.append(nome)
                    elif status == "serviço ruim":
                        st.warning(f"**{nome}**\n\n⚠️ {status.upper()}")
                        setores_criticos.append(nome)
                    elif status == "serviço bom":
                        st.info(f"**{nome}**\n\n👍 {status.upper()}")
                    else:
                        st.success(f"**{nome}**\n\n⭐ {status.upper()}")
                    
                    st.metric(label="Média Reclamações", value=f"{row['reclamacoes_mensais']:.1f}")
                    st.metric(label="Nota de Satisfação", value=f"{row['nota_satisfacao']:.1f}/5.0")

            st.write("---")
            
            # Recomendação Estratégica Direcionada para Empreendedores
            st.write("### 📈 Plano de Ação Recomendado pela IA")
            if setores_criticos:
                st.error("🚨 **Atenção Gestor:** Foi identificada necessidade severa de intervenção operacional.")
                st.markdown(f"Os seguintes serviços foram classificados abaixo do limite tolerável e **NECESSITAM DE TREINAMENTO IMEDIATO** das suas equipes:")
                for item in setores_criticos:
                    st.markdown(f"- **{item}**")
            else:
                st.success("✅ **Excelente!** Todas as áreas mapeadas pela inteligência artificial apresentam comportamento saudável e não necessitam de treinamentos de correção emergenciais.")
                
        except Exception as e:
            st.error(f"Erro Operacional no processamento: {str(e)}")
else:
    st.info("Aguardando ativação do sistema. Clique no botão acima para rodar a análise por Redes Neurais.")