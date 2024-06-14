FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install Flask \
    && pip install gunicorn \
    && pip install google-cloud-storage \
    && pip install pillow \
    && pip install numpy \
    && pip install tensorflow \
    && pip install keras

COPY . /app

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
