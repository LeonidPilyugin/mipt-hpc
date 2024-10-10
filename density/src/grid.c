#include <grid.h>
#include <stdlib.h>
#include <string.h>

Grid * grid_init(i_t x_size, i_t z_size, f_t xl, f_t xh, f_t zl, f_t zh) {
    Grid * result = malloc(sizeof(Grid));
    if (!result) return NULL;
    result->x_size = x_size;
    result->z_size = z_size;
    result->v = malloc(sizeof(f_t) * x_size * z_size);
    if (!result->v) return NULL;
    result->xl = xl;
    result->xh = xh;
    result->zl = zl;
    result->zh = zh;
    return result;
}

void grid_destroy(Grid * g) {
    free(g->v);
    free(g);
}

void grid_add(Grid * left, const Grid * right) {
    for (i_t i = 0; i < left->x_size; i++) {
        for (i_t j = 0; j < left->z_size; j++) {
            grid_element(left, i, j) += grid_element(right, i, j);
        }
    }
}

void grid_mul(Grid * left, f_t factor) {
    for (i_t i = 0; i < left->x_size; i++) {
        for (i_t j = 0; j < left->z_size; j++) {
            grid_element(left, i, j) *= factor;
        }
    }
}

void grid_write(const Grid * g, FILE * fp) {
    fprintf(fp, "x,y,v\n");
    for (i_t i = 0; i < g->x_size; i++) {
        for (i_t j = 0; j < g->z_size; j++) {
            fprintf(fp, "%d,%d,%f\n", i, j, grid_element(g, i, j));
        }
    }
}

void grid_reset(Grid * g) {
    memset(g->v, 0, g->z_size * g->x_size * sizeof(f_t));
}
