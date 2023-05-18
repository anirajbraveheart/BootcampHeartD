FROM python:3.11.1

WORKDIR /heart_D

EXPOSE 8501

COPY . /heart_D

RUN pip install -r requirements.txt

CMD streamlit run server.py