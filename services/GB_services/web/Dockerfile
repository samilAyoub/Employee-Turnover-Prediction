# pull official base image
FROM python:3.8.3-slim-buster

# set work directory
WORKDIR /usr/src/gb_app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/gb_app/requirements.txt
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/gb_app/