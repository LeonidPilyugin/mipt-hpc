#pragma once

#include <stdio.h>
#include <types.h>

#define SCALE_FACTOR 6.02e-7f

typedef struct {
    f_a x, y, z, m;
    i_t size;
} Particles;

Particles * particles_init(i_t size);
Particles * particles_init_csv(i_t size, FILE * fp);
void particles_destroy(Particles * p);
