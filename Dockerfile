FROM python:3.6.9-slim-buster
LABEL maintainer="antoniop.camargo@gmail.com"

RUN pip install --no-cache-dir 'rnasamba==0.1.5' 'biopython==1.74' \
    'numpy==1.16.5' 'keras==2.3.0' 'tensorflow==1.14.0'

VOLUME ["/app"]
WORKDIR /app
ENTRYPOINT ["rnasamba"]