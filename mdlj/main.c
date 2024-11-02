#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

const int periods[3] = { 1, 1, 1 };

typedef struct {
    int domens[3];
    double dt;
    char * output_dir;
    double lattice_parameter;
    int box_size[3];
    double temperature;
    int dump_every;
    int total_steps;
} Options;

Options * options_init() {
    Options * result = malloc(sizeof(Options));

    if (!result) goto error;

    for (int i = 0; i < 3; i++) result->domens[i] = 1;
    result->dt = 1e-3;
    result->output_dir = ".";
    result->lattice_parameter = 1.0;
    for (int i = 0; i < 3; i++) result->box_size[i] = 10;
    result->temperature = 1.0;
    result->dump_every = 10;
    result->total_steps = 1000;

    return result;

error:
    return NULL;
}

void options_destroy(Options * op) {
    free(op);
}

Options * parse_cla(int argc, char ** argv) {
    Options * result = options_init();
    char * endptr;
    if (!result) goto error;

    for (int i = 1; i < argc; i++) {
        if (!strcmp("-d", argv[i])) {
            for (int j = 0; j < 3; j++) {
                result->domens[j] = strtol(argv[++i], &endptr, 10);
                if (endptr == argv[i] || *endptr) goto error;
                if (result->domens[j] <= 0) goto error;
            }
        } else if (!strcmp("-dt", argv[i])) {
            result->dt = strtod(argv[++i], &endptr);
            if (endptr == argv[i] || *endptr) goto error;
            if (result->dt <= 0.0) goto error;
        } else if (!strcmp("-dir", argv[i])) {
            result->output_dir = argv[++i];
        } else if (!strcmp("-p", argv[i])) {
            result->lattice_parameter = strtod(argv[++i], &endptr);
            if (endptr == argv[i] || *endptr) goto error;
            if (result->lattice_parameter <= 0.0) goto error;
        } else if (!strcmp("-bx", argv[i])) {
            for (int j = 0; j < 3; j++) {
                result->box_size[j] = strtol(argv[++i], &endptr, 10);
                if (endptr == argv[i] || *endptr) goto error;
                if (result->box_size[j] <= 0) goto error;
            }
        } else if (!strcmp("-t", argv[i])) {
            result->temperature = strtod(argv[++i], &endptr);
            if (endptr == argv[i] || *endptr) goto error;
            if (result->lattice_parameter <= 0.0) goto error;
        } else if (!strcmp("-de", argv[i])) {
            result->dump_every = strtol(argv[++i], &endptr, 10); 
            if (endptr == argv[i] || *endptr) goto error;
            if (result->dump_every <= 0.0) goto error;
        } else if (!strcmp("-s", argv[i])) {
            result->total_steps = strtol(argv[++i], &endptr, 10); 
            if (endptr == argv[i] || *endptr) goto error;
            if (result->total_steps <= 0.0) goto error;
        } else goto error;
    }

    return result;

error:
    return NULL;
}


int init_mpi(int argc, char ** argv, const Options * options) {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1 + options->domens[0] * options->domens[1] * options->domens[2]) goto error;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
error:
    return -1;
}

void finalize_mpi() {
    MPI_Finalize();
}

typedef struct {
    int size;
    double * x, * y, * z,
           * vx, * vy, * vz,
           * fx, * fy, * fz, *U;
} Particles;

Particles * particles_create(int size) {
    Particles * result = malloc(sizeof(Particles));
    if (!result) goto error;

    result->size = size;
    result->x = malloc(sizeof(double) * size);
    result->y = malloc(sizeof(double) * size);
    result->z = malloc(sizeof(double) * size);
    result->vx = malloc(sizeof(double) * size);
    result->vy = malloc(sizeof(double) * size);
    result->vz = malloc(sizeof(double) * size);
    result->fx = malloc(sizeof(double) * size);
    result->fy = malloc(sizeof(double) * size);
    result->fz = malloc(sizeof(double) * size);
    result->U = malloc(sizeof(double) * size);

    if (!(
        result->x && result->y && result->z &&
        result->vx && result->vy && result->vz &&
        result->fx && result->fy && result->fz && result->U
    )) goto error;

    return result;

error:
    return NULL;
}

void particles_destroy(Particles * p) {
    free(p->x);
    free(p->y);
    free(p->z);
    free(p->vx);
    free(p->vy);
    free(p->vz);
    free(p->fx);
    free(p->fy);
    free(p->fz);
    free(p->U);
    free(p);
}

void master(const Options * op) {
    int size = op->box_size[0] * op->box_size[1] * op->box_size[2];
    Particles * all_particles = particles_create(size);

    // create all particles
    for (int i = 0; i < size; i++) {
        all_particles->x[i] = op->lattice_parameter * (i % op->box_size[0]);
        all_particles->y[i] = op->lattice_parameter * ((i / op->box_size[0]) % op->box_size[1]);
        all_particles->z[i] = op->lattice_parameter * ((i / (op->box_size[0] * op->box_size[1])) % op->box_size[2]);
        all_particles->vx[i] = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
        all_particles->vy[i] = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
        all_particles->vz[i] = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
        all_particles->fx[i] = all_particles->fy[i] = all_particles->fz[i] = 0;
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double * send_buf = malloc(sizeof(double) * 6 * size);
    double * recv_buf = malloc(sizeof(double) * 3 * size);
    double * energy_recv_buf = malloc(sizeof(double) * size * 2);

    memcpy(send_buf, all_particles->x, sizeof(double) * size);
    memcpy(send_buf + size, all_particles->y, sizeof(double) * size);
    memcpy(send_buf + size * 2, all_particles->z, sizeof(double) * size);
    memcpy(send_buf + size * 3, all_particles->vx, sizeof(double) * size);
    memcpy(send_buf + size * 4, all_particles->vy, sizeof(double) * size);
    memcpy(send_buf + size * 5, all_particles->vz, sizeof(double) * size);
    particles_destroy(all_particles);

    // send initial data
    MPI_Bcast(&send_buf, size * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double kinetic_energy, potential_energy;

    printf("step,T,U,E\n");
    fflush(stdout);

    char str_buf[4096];

    int counts[100], displs[100];

    for (int i = 0; i < op->total_steps; i += op->dump_every) {
        kinetic_energy = potential_energy = 0.0;
        MPI_Gatherv(NULL, 0, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(NULL, 0, MPI_DOUBLE, energy_recv_buf, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        for (int j = 0; j < size; j++) {
            kinetic_energy += energy_recv_buf[j];
            potential_energy += energy_recv_buf[j + size];
        }
        printf("%d,%lf,%lf,%lf\n", i, kinetic_energy, potential_energy, kinetic_energy + potential_energy);
        fflush(stdout);
        
        sprintf(str_buf, "%s/%d.atom", op->output_dir, i);
        
        FILE * out = fopen(str_buf, "w");

        fprintf(out, "ITEM: TIMESTEP\n%d\n", i);
        fprintf(out, "ITEM: NUMBER OF ATOMS\n%d\n", size);
        fprintf(out, "ITEM: BOX BOUNDS pp pp pp\n0.0 %lf\n0.0 %lf\n 0.0 %lf\n", op->box_size[0] * op->lattice_parameter, op->box_size[1] * op->lattice_parameter, op->box_size[2] * op->lattice_parameter);
        fprintf(out, "ITEM: ATOMS x y z\n");
        for (int j = 0; j < size; j++) {
            fprintf(out, "%lf %lf %lf\n", recv_buf[j], recv_buf[j + size], recv_buf[j + 2 * size]);
        }

        fclose(out);
    }

    free(send_buf);
    free(recv_buf);
    free(energy_recv_buf);
}

void slave(const Options * op) {
    // init topology
    MPI_Comm cart_comm;
    int world_rank, cart_rank, cart_coords[3];
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Cart_create(MPI_COMM_WORLD, 3, op->domens, periods, 0, &cart_comm);
    MPI_Cart_map(cart_comm, 3, op->domens, periods, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 3, cart_coords);

    int total_particles;
    MPI_Bcast(&total_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double * send_buf = malloc(sizeof(double) * 10 * total_particles);
    double * recv_buf = malloc(sizeof(double) * 60 * total_particles);
    double * energy_send_buf = malloc(sizeof(double) * 2 * total_particles);

    Particles * particles = particles_create(total_particles);
    // get initial data
    MPI_Bcast(recv_buf, 6 * total_particles, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double cell_bounds[6];
    for (int i = 0; i < 3; i++) {
        cell_bounds[i] = ((double) cart_coords[i]) / ((double) op->domens[i]) * op->lattice_parameter;
        cell_bounds[i+3] = cell_bounds[i] + op->lattice_parameter / ((double) op->domens[i]);
    }

    particles->size = 0;
    for (int i = 0; i < total_particles; i++) {
        if (
            cell_bounds[0] <= recv_buf[i] && cell_bounds[3] > recv_buf[i] &&
            cell_bounds[1] <= recv_buf[i+total_particles] && cell_bounds[4] > recv_buf[i+total_particles] &&
            cell_bounds[2] <= recv_buf[i+total_particles*2] && cell_bounds[5] > recv_buf[i+total_particles*2]
        ) {
            particles->x[particles->size] = recv_buf[i];
            particles->y[particles->size] = recv_buf[i+total_particles];
            particles->z[particles->size] = recv_buf[i+2*total_particles];
            particles->vx[particles->size] = recv_buf[i+3*total_particles];
            particles->vy[particles->size] = recv_buf[i+4*total_particles];
            particles->vz[particles->size] = recv_buf[i+5*total_particles];
            particles->fx[particles->size] = particles->fy[particles->size] = particles->fz[particles->size] = 0.0;
            particles->size++;
        }
    }

    int displs[6], counts[6];
    for (int step = 0; step < op->total_steps; step++) {
        memcpy(send_buf, particles->x, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size, particles->y, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 2, particles->z, sizeof(double) * particles->size);

        MPI_Neighbor_allgatherv(send_buf, particles->size * 3, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, cart_comm);

        // compute forces
        for (int i = 0; i < particles->size; i++) {
            particles->U[i] = 0.0;
            double distance, force, energy;
            particles->fx[i] = particles->fy[i] = particles->fz[i] = 0.0;
            for (int j = 0; j < particles->size; j++) {
                if (i == j) continue;
                distance = hypot(
                    fmod(particles->x[i] - particles->x[j], op->box_size[0] * op->lattice_parameter),
                    hypot(
                        fmod(particles->y[i] - particles->y[j], op->box_size[1] * op->lattice_parameter),
                        fmod(particles->z[i] - particles->z[j], op->box_size[2] * op->lattice_parameter)
                    )
                );
                if (distance > 2.5) continue;
                force = 48 * (pow(distance, 12) - 0.5 * pow(distance, 6));
                particles->fx[i] += fmod(particles->x[i] - particles->x[j], op->box_size[0] * op->lattice_parameter) / distance / distance * force;
                particles->fy[i] += fmod(particles->y[i] - particles->y[j], op->box_size[1] * op->lattice_parameter) / distance / distance * force;
                particles->fz[i] += fmod(particles->z[i] - particles->z[j], op->box_size[2] * op->lattice_parameter) / distance / distance * force;
                if (step % op->dump_every == 0) {
                    particles->U[i] += 4 * (pow(distance, 12) - pow(distance, 6));
                }
            }

            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < counts[i] / 3; k++) {
                    int index = displs[i] + k;
                    distance = hypot(
                        fmod(particles->x[i] - recv_buf[index], op->box_size[0] * op->lattice_parameter),
                        hypot(
                            fmod(particles->y[i] - recv_buf[index + counts[i] / 3], op->box_size[1] * op->lattice_parameter),
                            fmod(particles->z[i] - recv_buf[index + 2 * counts[i] / 3], op->box_size[2] * op->lattice_parameter)
                        )
                    );
                    if (distance > 2.5) continue;
                    force = 48 * (pow(distance, 12) - 0.5 * pow(distance, 6));
                    particles->fx[i] += fmod(particles->x[i] - recv_buf[index], op->box_size[0] * op->lattice_parameter) / distance / distance * force;
                    particles->fy[i] += fmod(particles->y[i] - recv_buf[index + counts[i] / 3], op->box_size[1] * op->lattice_parameter) / distance / distance * force;
                    particles->fz[i] += fmod(particles->z[i] - recv_buf[index + 2 * counts[i] / 3], op->box_size[2] * op->lattice_parameter) / distance / distance * force;
                    if (step % op->dump_every == 0) {
                        particles->U[i] += 4 * (pow(distance, 12) - pow(distance, 6));
                    }
                }
            }
            particles->vx[i] += particles->fx[i] * op->dt;
            particles->vy[i] += particles->fy[i] * op->dt;
            particles->vz[i] += particles->fz[i] * op->dt;
            particles->x[i] += particles->vx[i] * op->dt;
            particles->y[i] += particles->vy[i] * op->dt;
            particles->z[i] += particles->vz[i] * op->dt;
        }

        memcpy(send_buf, particles->x, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size, particles->y, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 2, particles->z, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 3, particles->vx, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 4, particles->vy, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 5, particles->vz, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 6, particles->fx, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 7, particles->fy, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 8, particles->fz, sizeof(double) * particles->size);
        memcpy(send_buf + particles->size * 9, particles->U, sizeof(double) * particles->size);

        printf("lol\n"); fflush(stdout);

        // update particles
        MPI_Neighbor_allgatherv(send_buf, particles->size * 10, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, cart_comm);

        int max_index = 0;
        for (int i = 0; i < particles->size; i++) {
            if (
                cell_bounds[0] <= fmod(particles->x[i], op->box_size[0] * op->lattice_parameter) && cell_bounds[3] > fmod(particles->x[i], op->box_size[0] * op->lattice_parameter) &&
                cell_bounds[1] <= fmod(particles->y[i], op->box_size[1] * op->lattice_parameter) && cell_bounds[4] > fmod(particles->y[i], op->box_size[1] * op->lattice_parameter) &&
                cell_bounds[2] <= fmod(particles->y[i], op->box_size[2] * op->lattice_parameter) && cell_bounds[5] > fmod(particles->y[i], op->box_size[2] * op->lattice_parameter)
            ) {
                particles->x[max_index] = particles->x[i];
                particles->y[max_index] = particles->y[i];
                particles->z[max_index] = particles->z[i];
                particles->vx[max_index] = particles->vx[i];
                particles->vy[max_index] = particles->vy[i];
                particles->vz[max_index] = particles->vz[i];
                particles->fx[max_index] = particles->fx[i];
                particles->fy[max_index] = particles->fy[i];
                particles->fz[max_index] = particles->fz[i];
                particles->U[max_index] = particles->U[i];
                max_index++;
            }
        }

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < counts[i] / 10; j++) {
                int index = displs[i] + j;
                double x, y, z;
                x = recv_buf[index];
                y = recv_buf[index + counts[i] / 10];
                z = recv_buf[index + counts[i] / 10 * 2];
                if (
                    cell_bounds[0] <= fmod(x, op->box_size[0] * op->lattice_parameter) && cell_bounds[3] > fmod(x, op->box_size[0] * op->lattice_parameter) &&
                    cell_bounds[1] <= fmod(y, op->box_size[1] * op->lattice_parameter) && cell_bounds[4] > fmod(y, op->box_size[1] * op->lattice_parameter) &&
                    cell_bounds[2] <= fmod(y, op->box_size[2] * op->lattice_parameter) && cell_bounds[5] > fmod(y, op->box_size[2] * op->lattice_parameter)
                ) {
                    particles->x[max_index] = x;
                    particles->y[max_index] = y;
                    particles->z[max_index] = z;
                    particles->vx[max_index] = recv_buf[index + counts[i] / 10 * 3];
                    particles->vy[max_index] = recv_buf[index + counts[i] / 10 * 4];
                    particles->vz[max_index] = recv_buf[index + counts[i] / 10 * 5];
                    particles->fx[max_index] = recv_buf[index + counts[i] / 10 * 6];
                    particles->fy[max_index] = recv_buf[index + counts[i] / 10 * 7];
                    particles->fz[max_index] = recv_buf[index + counts[i] / 10 * 8];
                    particles->U[max_index] = recv_buf[index + counts[i] / 10 * 9];

                    max_index++;
                }
            }
        }

        particles->size = max_index;
        
        if (step % op->dump_every == 0) {
            memcpy(send_buf, particles->x, sizeof(double) * particles->size);
            memcpy(send_buf + particles->size, particles->y, sizeof(double) * particles->size);
            memcpy(send_buf + particles->size * 2, particles->z, sizeof(double) * particles->size);

            for (int i = 0; i < particles->size; i++) {
                energy_send_buf[i] += 0.5 * hypot(hypot(particles->vx[i], particles->vy[i]), particles->vz[i]);
                energy_send_buf[i + particles->size] = particles->U[i];
            }

            printf("Sending particles\n");
            MPI_Gatherv(send_buf, 3 * particles->size, MPI_DOUBLE, NULL, 0, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("Sending energies\n");
            MPI_Gatherv(energy_send_buf, particles->size * 2, MPI_DOUBLE, NULL, 0, NULL, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    particles_destroy(particles);
    free(send_buf);
    free(recv_buf);
    free(energy_send_buf);
}

int main(int argc, char * argv[]) {
    Options * options = parse_cla(argc, argv);
    if (!options) goto error;

    struct stat st = {0};
    if (stat(options->output_dir, &st) == -1 && !mkdir(options->output_dir, 0777)) goto error;

    int rank = init_mpi(argc, argv, options);
    if (-1 == rank) goto error;

    if (rank) slave(options); else master(options);

    options_destroy(options);
    finalize_mpi();

    return 0;

error:
    return 1;
}
