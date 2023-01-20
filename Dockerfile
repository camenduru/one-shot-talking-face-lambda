# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.1/ubuntu2004/devel/cudnn8/Dockerfile
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /content
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y sudo && apt-get install -y python3-pip && pip3 install --upgrade pip
RUN apt-get install -y gnupg wget htop sudo git git-lfs software-properties-common build-essential cmake curl
RUN apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libgl1 libgtk2.0-0 jq libdc1394-22-dev libraw1394-dev libopenblas-base

ENV PATH="/home/admin/.local/bin:${PATH}"

RUN pip3 install pandas scipy matplotlib torch torchvision torchaudio gradio altair imageio-ffmpeg pocketsphinx jq "numpy<1.24"

RUN git lfs install
RUN git clone https://huggingface.co/camenduru/pocketsphinx-20.04-t4 pocketsphinx && cd pocketsphinx && cmake --build build --target install

RUN git clone https://huggingface.co/camenduru/one-shot-talking-face-20.04-t4 one-shot-talking-face && cd one-shot-talking-face && pip install -r requirements.txt && chmod 755 OpenFace/FeatureExtraction
RUN mkdir /content/out

COPY app.py /content/app.py
COPY examples /content/examples

RUN adduser --disabled-password --gecos '' admin
RUN adduser admin sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN chown -R admin:admin /content
RUN chmod -R 777 /content
RUN chown -R admin:admin /home
RUN chmod -R 777 /home
USER admin

EXPOSE 7860

CMD ["python3", "app.py"]