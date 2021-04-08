FROM python:3.8

WORKDIR /chatbot
ADD . /chatbot

RUN pip3 install --no-cache-dir -r requirements.txt

# Rebuild numpy and gensim to address https://github.com/RaRe-Technologies/gensim/issues/2309
RUN pip3 uninstall -y numpy
RUN pip3 uninstall -y gensim
RUN pip3 install numpy==1.19.5
RUN pip3 install gensim==3.6.0

EXPOSE 22370
CMD ["python", "main.py", "serve"]
