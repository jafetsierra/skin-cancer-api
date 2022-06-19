FROM python:3.9-slim
ENV PORT=8000
COPY requirements.txt /
RUN pip install requirements.txt
COPY ./app /app
COPY ./model/SkinCancer_model_v5.h5

ENTRYPOINT uvicorn app.main:app --host 0.0.0.0 --port $PORT