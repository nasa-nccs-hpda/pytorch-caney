# Arguments to pass to the image
ARG VERSION_DATE=24.01
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

# System dependencies
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libfontconfig1 \
        bzip2 \
        diffutils \
        file \
        build-essential \
        make \
        swig \
        libnetcdf-dev \
        libacl1-dev \
        libgeos++-dev \
        libgeos-dev \
        libsqlite3-dev \
        libx11-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        libspatialindex-dev \
        wget \
        vim \
        curl \
        git \
        procps \
        gcc \
        g++ \
        bzip2 \
        libssl-dev \
        libzmq3-dev \
        libpng-dev \
        libfreetype6-dev \ 
        locales \
        git-lfs && \
    apt-get -y install gdal-bin libgdal-dev && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt /var/lib/apt/lists/*

# Install shiftc
WORKDIR /app
RUN git clone --single-branch --branch master https://github.com/pkolano/shift.git && \
    cd shift/c && \
    make nolustre && \
    cd ../ && \
    install -m 755 perl/shiftc /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    install -m 755 perl/shift-mgr /usr/local/bin/ && \
    install -m 644 etc/shiftrc /etc/ && \
    install -m 755 perl/shift-aux /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    export LC_ALL=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    locale-gen en_US.UTF-8 && \
    rm -rf /app

# Pip
RUN pip --no-cache-dir install \
    pytorch-lightning \
    Lightning \
    transformers \
    datasets \
    deepspeed \
    webdataset \
    'huggingface_hub[cli,torch]' \
    torchgeo \
    rasterio \
    rioxarray \
    xarray \
    xarray-spatial \
    geopandas \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    tifffile \
    webcolors \
    tiler \
    segmentation-models \
    timm \
    supervision \
    pytest \
    coveralls \
    rtree \
    sphinx \
    sphinx_rtd_theme \
    yacs \
    termcolor \
    segmentation-models-pytorch \
    coverage

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
