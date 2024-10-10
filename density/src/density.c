#include <math.h>
#include <density.h>

#define PI 3.1415926f

f_t origin_kernel(f_t r, f_t h) {
    r /= h;
    if (r >= 2.f) return 0.f;
    if (r >= 1.f) return (2.f-r)*(2.f-r)*(2.f-r) / 4.f / PI / h / h / h;
    return (1.f - 1.5f * r * r + 0.75f * r * r * r) / PI / h / h / h;
}

f_t kernel(f_t r2, f_t h) {
    r2 /= h * h;
    if (r2 >= 2.f) return 0.f;
    if (r2 >= 1.f) return (2.f-r2)*(2.f-r2) * 0.15 / PI / h / h / h;
    return (1.f - 1.56f * r2 + 0.7f * r2 * r2) / PI / h / h / h;
}

void get_density(Particles * p, Grid * g, DensityParams * params) {
    f_t dist = 0.f;
    for (i_t i = 0; i < p->size; i++) {
        for (i_t x = 0; x < g->x_size; x++) {
            for (i_t z = 0; z < g->z_size; z++) {
                dist = sqrtf((grid_x(g, x) - p->x[i]) * (grid_x(g, x) - p->x[i]) + (grid_z(g, z) - p->z[i]) * (grid_z(g, z) - p->z[i]) + (p->y[i]) * (p->y[i]));
                grid_element(g, x, z) += params->m * origin_kernel(dist, params->h);

                /* dist = (grid_x(g, x) - p->x[i]) * (grid_x(g, x) - p->x[i]) + (grid_z(g, z) - p->z[i]) * (grid_z(g, z) - p->z[i]) + (p->y[i]) * (p->y[i]); */
                /* grid_element(g, x, z) += params->m * kernel(dist, params->h); */
            }
        }
    }
}
