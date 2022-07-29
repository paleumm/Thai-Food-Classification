FROM python:3.10.4

WORKDIR /app

COPY app.py ./app.py
COPY class.txt ./class.txt
COPY model_pretrained_True.pth ./model_pretrained_True.pth
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]