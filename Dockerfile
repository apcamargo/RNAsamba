FROM python:3.6.9-slim-buster
LABEL maintainer="antoniop.camargo@gmail.com"

RUN pip install --no-cache-dir \
    'biopython==1.74' \
    'keras==2.3.0' \
    'numpy==1.16.5' \
    'rnasamba==0.1.6' \
    'tensorflow==1.14.0'

VOLUME ["/app"]
WORKDIR /app
ENTRYPOINT ["rnasamba"]