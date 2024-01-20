FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

#RUN apt clean
RUN apt update
RUN #apt install nvidia-driver-535 -y
RUN apt install git meson libluajit-5.1-dev -y
RUN apt install libavcodec-dev libavfilter-dev libplacebo-dev libass-dev -y

COPY ./audio /tmp/mpv/audio
COPY ./buildDir /tmp/mpv/buildDir
COPY ./ci /tmp/mpv/ci
COPY ./DOCS /tmp/mpv/DOCS
COPY ./etc /tmp/mpv/etc
COPY ./common /tmp/mpv/common
COPY ./demux /tmp/mpv/demux
COPY ./filters /tmp/mpv/filters
COPY ./input /tmp/mpv/input
COPY ./libmpv /tmp/mpv/libmpv
COPY ./misc /tmp/mpv/misc
COPY ./options /tmp/mpv/options
COPY ./osdep /tmp/mpv/osdep
COPY ./player /tmp/mpv/player
COPY ./stream /tmp/mpv/stream
COPY ./sub /tmp/mpv/sub
COPY ./ta /tmp/mpv/ta
COPY ./test /tmp/mpv/test
COPY ./TOOLS /tmp/mpv/TOOLS
COPY ./video /tmp/mpv/video
COPY ./meson.build /tmp/mpv/meson.build
COPY ./meson_options.txt /tmp/mpv/meson_options.txt
COPY ./mpv_talloc.h /tmp/mpv/mpv_talloc.h
COPY ./VERSION /tmp/mpv/VERSION


WORKDIR /tmp/mpv
RUN apt remove meson -y
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip3 install --user meson
RUN apt install wget -y
RUN git clone --recursive https://code.videolan.org/videolan/libplacebo

#ARG SOURCE_DIR=/tmp/shaderc
#
#RUN git clone https://github.com/google/shaderc $SOURCE_DIR
#RUN apt install cmake -y
#RUN cd $SOURCE_DIR && ./utils/git-sync-deps
#RUN cd $BUILD_DIR && cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo $SOURCE_DIR && ninja
RUN apt install  glslang-dev  -y
#RUN wget https://launchpad.net/ubuntu/+source/spirv-tools/2022.1+1.3.204.0-2/+build/23175149/+files/spirv-tools_2022.1+1.3.204.0-2_amd64.deb && apt install -y ./spirv-tools_2022.1+1.3.204.0-2_amd64.deb
#RUN wget https://launchpad.net/ubuntu/+source/glslang/11.8.0+1.3.204.0-1/+build/23173633/+files/glslang-dev_11.8.0+1.3.204.0-1_amd64.deb && apt install -y ./glslang-dev_11.8.0+1.3.204.0-1_amd64.deb
#RUN wget http://launchpadlibrarian.net/647824148/libshaderc1_2023.2-1_amd64.deb && apt install -y ./libshaderc1_2023.2-1_amd64.deb
#RUN wget https://launchpad.net/ubuntu/+source/shaderc/2023.2-1/+build/25518458/+files/glslc_2023.2-1_amd64.deb && apt install -y ./glslc_2023.2-1_amd64.deb
#RUN wget http://launchpadlibrarian.net/647824145/libshaderc-dev_2023.2-1_amd64.deb && apt install -y ./libshaderc-dev_2023.2-1_amd64.deb
#
#RUN wget http://launchpadlibrarian.net/694063520/libc6_2.38-3ubuntu1_amd64.deb && apt install -y ./libc6_2.38-3ubuntu1_amd64.deb
#RUN wget https://launchpad.net/ubuntu/+source/libplacebo/6.338.1-2/+build/26968818/+files/libplacebo338_6.338.1-2_amd64.deb && apt install -y ./libplacebo338_6.338.1-2_amd64.deb
#RUN wget https://launchpad.net/ubuntu/+source/libplacebo/6.338.1-2/+build/26968818/+files/libplacebo338-dbgsym_6.338.1-2_amd64.ddeb && apt install -y ./libplacebo338-dbgsym_6.338.1-2_amd64.ddeb
#RUN wget https://launchpad.net/ubuntu/+source/libplacebo/6.338.1-2/+build/26968818/+files/libplacebo-dev_6.338.1-2_amd64.deb && apt install -y ./libplacebo-dev_6.338.1-2_amd64.deb
RUN apt install ninja-build pkg-config libvulkan-dev -y
#RUN cd libplacebo && \
#    DIR=./build && \
#    ~/.local/bin/meson setup $DIR -Dshaderc=enabled -Dvulkan=enabled && \
#    ~/.local/bin/meson compile -C $DIR && \
#    ~/.local/bin/meson install -C ./build
#

RUN cd libplacebo && ~/.local/bin/meson build -Dshaderc=disabled && ninja -C build && ninja -C build install

RUN ~/.local/bin/meson setup build
RUN ~/.local/bin/meson compile -C build
RUN ~/.local/bin/meson install -C build

WORKDIR /home
RUN rm -rf /tmp/mpv
RUN apt remove meson -y
ENTRYPOINT ["mpv","--vf=gpu=api=help"]