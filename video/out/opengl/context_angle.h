#pragma once

#include <stdbool.h>

struct m_sub_options;

struct angle_opts {
    int renderer;
    int d3d11_warp;
    int d3d11_feature_level;
    int egl_windowing;
    bool flip;
};

extern const struct m_sub_options angle_conf;
