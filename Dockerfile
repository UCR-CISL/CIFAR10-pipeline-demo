FROM python:3.8-slim

WORKDIR /model

COPY . /model

RUN pip install --trusted-host pypi.python.org -r requirements.txt

#EXPOSE 80

CMD ["python", "main_build.py"]
