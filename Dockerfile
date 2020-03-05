FROM dre_hh/spark
WORKDIR /app

COPY bin /app/bin
COPY requirements.txt /app
RUN python3 -m pip install numpy
RUN python3 -m pip install jupyter

