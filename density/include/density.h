#pragma once

#include <grid.h>
#include <particles.h>

typedef struct {
    f_t m, h;
} DensityParams;

void get_density(Particles * p, Grid * g, DensityParams * params);
