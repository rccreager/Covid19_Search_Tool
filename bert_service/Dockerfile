FROM nvidia/cuda:10.2-base-ubuntu18.04 

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip python3-venv wget unzip vim 

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install --upgrade pip 
RUN pip3 install bert-serving-server==1.10 
RUN pip3 install bert-serving-client==1.10.0
RUN pip3 install tensorflow-gpu==1.15.2

RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
RUN unzip uncased_L-12_H-768_A-12.zip
RUN rm uncased_L-12_H-768_A-12.zip

COPY ./ /bert 
RUN ["chmod", "+x", "/bert/entrypoint.sh"]
ENTRYPOINT ["/bert/entrypoint.sh"]
CMD []

#CMD bert-serving-start -model_dir /uncased_L-12_H-768_A-12/ -num_worker=2 -port_out=5556
