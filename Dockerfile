FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#COPY app app

#COPY data/raw/data_raw.csv /app/data/raw/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]
