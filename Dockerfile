FROM python:3.9.13

# For some reason lib boost has problems with 3.10 so this image is 3.9.13.
ARG BOOST_URL=https://sourceforge.net/projects/boost/files/boost/1.74.0/boost_1_74_0.tar.bz2/download

COPY . .
WORKDIR /spnn
ENV PYTHON_PATH=/root

RUN apt-get update -qq && \
    apt-get upgrade -yqq && \
    apt-get install -yqq --install-recommends \
        libarmadillo-dev \
        libboost-numpy-dev

RUN wget -O /usr/boost_1_74_0.tar.bz2 $BOOST_URL && \
    cd /usr/ && \
    tar --bzip2 -xf /usr/boost_1_74_0.tar.bz2 && \
    cd /usr/boost_1_74_0 && \
    ./bootstrap.sh --with-python=python3 --prefix=/usr && \
    ./b2 install --with-python

RUN pip3 install --upgrade pip && \
    pip3 install \
        --root-user-action=ignore \
        --no-cache-dir \
        --upgrade \
        -r requirements.txt

# Run tests for Python and Boost C++ extensions that were built from setup.py.
RUN python3 -m unittest -v test.test_spnn