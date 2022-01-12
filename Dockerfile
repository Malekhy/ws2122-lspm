FROM python:3.7.9   

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install -r /requirements.txt

RUN mkdir -p /ws2122-lspm/media/event_logs/
COPY ./ws2122-lspm /ws2122-lspm
WORKDIR /ws2122-lspm
COPY ./scripts /scripts

RUN chmod +x /scripts/*

RUN mkdir -p /media
RUN mkdir -p /static

ADD ./ws2122-lspm ws2122-lspm/
RUN mkdir /media/event_logs/
RUN chmod -R 777 /media/event_logs/
RUN ls -ltr




CMD ["entrypoint.sh"]







