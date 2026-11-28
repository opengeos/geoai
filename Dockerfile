FROM quay.io/jupyter/base-notebook:python-3.13

# -------------------------------------------------------
# 1. Install system-level packages (minimal, just git)
# -------------------------------------------------------
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 2. Install binary geospatial dependencies via conda (base env)
# -------------------------------------------------------
RUN mamba install -n base -c conda-forge \
    gdal \
    geopandas \
    rasterio \
    overturemaps -y && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}"

# -------------------------------------------------------
# 3. Environment variables
# -------------------------------------------------------
ENV PROJ_LIB=/opt/conda/share/proj
ENV GDAL_DATA=/opt/conda/share/gdal
ENV LOCALTILESERVER_CLIENT_PREFIX='proxy/{port}'

# -------------------------------------------------------
# 4. Copy source code (do this *after* package installs to improve caching)
# -------------------------------------------------------
COPY . /home/jovyan/geoai
WORKDIR /home/jovyan/geoai

# -------------------------------------------------------
# 5. Build and install geoai from source
# -------------------------------------------------------
RUN rm -rf /home/jovyan/geoai/geoai.egg-info && \
    pip install --no-cache-dir . && \
    mkdir -p /home/jovyan/work && \
    fix-permissions /home/jovyan

# -------------------------------------------------------
# 6. Set back to default user
# -------------------------------------------------------
WORKDIR /home/jovyan
USER jovyan

# -------------------------------------------------------
# 7. Run the docker container
# -------------------------------------------------------
# docker run -it -p 8888:8888 -v $(pwd):/home/jovyan/work giswqs/geoai:latest
