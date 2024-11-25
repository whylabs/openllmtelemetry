FROM python:3.9-slim

WORKDIR /app

RUN apt update && apt install -y git

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]