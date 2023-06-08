# Arguments to pass to the image
ARG VERSION_DATE=23.01
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH="/mnt/data-store/FileGDB_API-64gcc51/lib/:/usr/local/lib"
ENV PROJ_LIB="/usr/share/proj"
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

# System dependencies
# libproj-dev gdal-bin libgdal-dev
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntugis/ppa && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential libsm6 libxext6 libxrender-dev libfontconfig1 \
        bzip2 diffutils file build-essential make swig wget libnetcdf-dev \
        libacl1-dev libgeos++-dev libgeos-dev libsqlite3-dev libx11-dev libproj-dev proj-data proj-bin \
        libspatialindex-dev wget vim curl git procps gcc g++ bzip2 libssl-dev locales && \
        apt-get -y install gdal-bin libgdal-dev git-lfs && \
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
RUN pip --no-cache-dir install omegaconf \
    pytorch-lightning \
    Lightning \
    transformers \
    datasets \
    'huggingface_hub[cli,torch]' \
    rasterio \
    rioxarray \
    xarray \
    geopandas \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    tifffile \
    webcolors \
    Pillow \
    seaborn \
    xgboost \
    tiler \
    segmentation-models \
    timm \
    supervision \
    pytest \
    coveralls \
    rtree \
    sphinx \
    sphinx_rtd_theme \
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'`

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
