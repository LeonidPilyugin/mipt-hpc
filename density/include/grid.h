#pragma once

#include <types.h>
#include <stdio.h>

typedef struct {
    i_t x_size, z_size;
    f_a v;
    f_t xl, xh, zl, zh;
} Grid;

Grid * grid_init(i_t x_size, i_t z_size, f_t xl, f_t xh, f_t zl, f_t zh);
void grid_destroy(Grid * g);

#define grid_element(g, x, z) (g)->v[(x) * (g)->z_size + (z)]
#define grid_x(g, x) ((g)->xl + ((g)->xh-(g)->xl) * x / g->x_size)
#define grid_z(g, z) ((g)->zl + ((g)->zh-(g)->zl) * z / g->z_size)

void grid_add(Grid * left, const Grid * right);
void grid_mul(Grid * left, f_t factor);

void grid_reset(Grid * g);

void grid_write(const Grid * g, FILE * fp);
