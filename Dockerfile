# Use a imagem oficial do Python
FROM python:3.11

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo requirements.txt e instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o diretório do aplicativo
#COPY app app

# Copie o arquivo data_raw.csv para o diretório /app/data/raw no contêiner
COPY data/raw/data_raw.csv /app/data/raw/

# Exponha a porta que o Streamlit usa (por padrão, é 8501)
EXPOSE 8501

# Comando para iniciar o aplicativo Streamlit quando o contêiner for iniciado
CMD ["streamlit", "run", "--server.enableCORS", "false", "--server.runOnSave", "true", "dashboard.py"]
