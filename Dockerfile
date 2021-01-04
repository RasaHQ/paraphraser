FROM ubuntu:20.04

RUN apt-get update

RUN apt-get install -y curl wget build-essential

RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash miniconda.sh -b \
    && rm -f miniconda.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /home

COPY download_model.sh download_model.sh
RUN ./download_model.sh

COPY . .

RUN conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
RUN pip install -r requirements.txt

RUN tar -xvf m39v1.tar

RUN touch run_paraphraser.py
