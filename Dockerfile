FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip python3-venv wget unzip vim 

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /
ADD /data/CORD-19-research-challenge/CORD-19-research-challenge.tar.gz /data/CORD-19-research-challenge/ 

RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt')"
RUN python3 -c "import nltk; nltk.download('stopwords')"
RUN python3 -c "import nltk; nltk.download('wordnet')"

RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
RUN unzip uncased_L-12_H-768_A-12.zip
RUN pip3 install bert-serving-server==1.10 --no-deps
RUN rm uncased_L-12_H-768_A-12.zip


