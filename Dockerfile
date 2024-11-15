FROM python:3.11
LABEL authors="maziar"

COPY src/ src/
WORKDIR src/

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt

EXPOSE 8080