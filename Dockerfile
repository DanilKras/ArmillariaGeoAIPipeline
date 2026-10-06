FROM apache/airflow:2.10.0-python3.12

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY airflow-requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt