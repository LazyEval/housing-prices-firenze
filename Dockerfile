FROM python:3.8-slim-buster
WORKDIR /usr/app/
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["src/deployment/deploy_model.py"]
