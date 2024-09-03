
FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api api

CMD uvicorn api.fast:app -- host 0.0.0.0 -- port $PORT
