
FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
