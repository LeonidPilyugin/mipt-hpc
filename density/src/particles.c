#include <particles.h>
#include <stdlib.h>
#include <string.h>

#define BUFSIZE 4096
#define DELIMETER ","

Particles * particles_init(i_t size) {
    Particles * result = malloc(sizeof(Particles));
    if (!result) return NULL;
    result->m = malloc(sizeof(f_t) * size);
    result->x = malloc(sizeof(f_t) * size);
    result->y = malloc(sizeof(f_t) * size);
    result->z = malloc(sizeof(f_t) * size);
    result->size = size;
    if (!(result->m && result->x && result->y && result->z)) return NULL;
    return result;
}

Particles * particles_init_csv(i_t size, FILE * fp) {
    Particles * result = particles_init(size);
    if (!result) return NULL;

    static char line[BUFSIZE];
    char * tokens[5];
    
    i_t line_n = 1;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp)) {
        tokens[0] = strtok(line, DELIMETER);      
        for (i_t i = 1; i < 5; i++) tokens[i] = strtok(NULL, DELIMETER);
        result->x[line_n - 1] = atof(tokens[0]) * SCALE_FACTOR;
        result->y[line_n - 1] = atof(tokens[1]) * SCALE_FACTOR;
        result->z[line_n - 1] = atof(tokens[2]) * SCALE_FACTOR;
        result->m[line_n - 1] = atof(tokens[4]);
        line_n++;
    }
    return result;
}

void particles_destroy(Particles * p) {
    free(p->m);    
    free(p->x);    
    free(p->y);    
    free(p->z);    
    free(p);
}
