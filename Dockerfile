FROM astrocrpublic.azurecr.io/runtime:3.0-1
FROM python:3.8-slim
RUN pip install mlflow
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow", "--host", "0.0.0.0"]