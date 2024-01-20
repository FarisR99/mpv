FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_THREADS=1

# Install dependencies
RUN apt update && apt upgrade -y
RUN apt install -y wget git libluajit-5.1-dev
RUN apt install -y libavcodec-dev libavfilter-dev libass-dev
RUN apt install -y libvulkan-dev liblcms2-dev

# Install python & meson
RUN apt purge -y meson || true
RUN apt install -y python3 python3-pip
RUN pip3 install --user meson
ENV ORIG_PATH="${PATH}"
ENV PATH="${PATH}:/root/.local/bin"

# Build MPV dependencies and MPV
RUN apt install -y dos2unix
COPY . /tmp/mpv
WORKDIR /tmp/mpv
RUN dos2unix **/* || true

## Build libplacebo
RUN git clone --recursive https://code.videolan.org/videolan/libplacebo
RUN apt install -y glslang-dev
RUN apt install -y ninja-build pkg-config libvulkan-dev
RUN cd libplacebo && \
    meson setup build -Ddemos=false -Dshaderc=disabled && \
    ninja -Cbuild -j${BUILD_THREADS} && \
    ninja -Cbuild install

## Build mpv
RUN meson setup build && \
    ninja -Cbuild -j${BUILD_THREADS} && \
    ninja -Cbuild install
### Verify
RUN ldconfig
RUN mpv -v

# Cleanup
WORKDIR /root
RUN rm -rf /tmp/mpv
RUN apt purge -y python3 python3-pip
ENV PATH="${ORIG_PATH}"
ENV ORIG_PATH=""

ENTRYPOINT ["mpv", "--vf=gpu=api=help"]
