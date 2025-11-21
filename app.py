import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuração da página
st.set_page_config(page_title="Sistema de Recomendação", page_icon="🎯")

st.title("🎯 Sistema de Recomendação de Projetos")
st.write("**Desenvolvido por Verônica Bergelino**")

st.info("""
Sistema de recomendação baseado em Machine Learning que sugere projetos 
alinhados com suas habilidades e interesses técnicos.
""")

# Dados dos projetos
projects_data = {
    'title': [
        'Chatbot RAG com LangChain',
        'Sistema de Recomendação', 
        'Análise de Sentimentos com NLP',
        'Dashboard de Analytics',
        'API REST com FastAPI',
        'App Mobile com React Native',
        'Sistema de E-commerce',
        'Plataforma de Cursos Online'
    ],
    'description': [
        'Chatbot inteligente usando retrieval-augmented generation e FAISS',
        'Sistema de recomendação baseado em conteúdo com machine learning',
        'Classificação de sentimentos usando transformers e processamento de linguagem',
        'Dashboard interativo para análise de dados empresariais',
        'API moderna com autenticação JWT e documentação automática',
        'Aplicativo mobile multiplataforma com React Native',
        'Loja virtual completa com carrinho e pagamentos',
        'Plataforma de ensino com vídeos, quizzes e certificados'
    ],
    'technologies': [
        'Python,LangChain,OpenAI,FAISS,Streamlit',
        'Python,Scikit-learn,Pandas,Streamlit,ML',
        'Python,Transformers,HuggingFace,NLP,Pytorch',
        'Python,Plotly,Dash,Pandas,SQL',
        'Python,FastAPI,SQLAlchemy,JWT,Swagger',
        'JavaScript,React Native,Node.js,Firebase',
        'JavaScript,React,Node.js,MongoDB,Stripe',
        'JavaScript,React,Node.js,MongoDB,AWS'
    ],
    'difficulty': ['Avançado', 'Intermediário', 'Avançado', 'Intermediário', 'Intermediário', 'Intermediário', 'Avançado', 'Avançado']
}

df = pd.DataFrame(projects_data)

# Interface
st.subheader("🔍 Encontre seu próximo projeto!")
user_skills = st.text_input("Digite suas habilidades (ex: Python, Machine Learning, React, JavaScript):")

if user_skills:
    # Combinar features para TF-IDF
    df['content'] = df['description'] + ' ' + df['technologies']
    
    # Vectorização
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Vectorizar input do usuário
    user_vector = tfidf.transform([user_skills])
    
    # Calcular similaridade
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix)
    
    # Recomendações
    similar_indices = cosine_sim.argsort()[0][-3:][::-1]
    
    st.subheader("📋 Projetos Recomendados Para Você:")
    for idx in similar_indices:
        if cosine_sim[0][idx] > 0:
            with st.container():
                st.write(f"### 🚀 {df.iloc[idx]['title']}")
                st.write(f"**Descrição:** {df.iloc[idx]['description']}")
                st.write(f"**Tecnologias:** {df.iloc[idx]['technologies']}")
                st.write(f"**Dificuldade:** {df.iloc[idx]['difficulty']}")
                st.write("---")
else:
    st.write("👆 Digite suas habilidades acima para receber recomendações personalizadas!")

st.write("---")
st.write("📧 **Contato**: veronica.bergelino@hotmail.com")
st.write("💼 **LinkedIn**: linkedin.com/in/veronica-bergelino")
