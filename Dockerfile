FROM python:3.8

WORKDIR /opt/silly-app

ENV PYTHONUNBUFFERED 1

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY tripadvisor_hotel_reviews.csv .

CMD ["python", "/opt/silly-app/app.py"]
