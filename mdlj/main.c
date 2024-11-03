#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

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

    MPI_Comm comm;
    int rank, cart_rank, cart_coords[3], master_rank;
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

    result->master_rank = result->domens[0] * result->domens[1] * result->domens[2];

    return result;

error:
    return NULL;
}


int init_mpi(int argc, char ** argv, Options * options) {
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1 + options->domens[0] * options->domens[1] * options->domens[2]) goto error;
    MPI_Comm_rank(MPI_COMM_WORLD, &(options->rank));
    MPI_Cart_create(MPI_COMM_WORLD, 3, options->domens, periods, 0, &(options->comm));
    if (MPI_COMM_NULL == options->comm) { assert(options->rank == options->master_rank); return options->rank; }
    MPI_Cart_map(options->comm, 3, options->domens, periods, &options->cart_rank);
    MPI_Cart_coords(options->comm, options->cart_rank, 3, options->cart_coords);
    return options->rank;
error:
    return -1;
}

void finalize_mpi() {
    MPI_Finalize();
}

#define MASTER_BEGIN if (op->rank == op->master_rank) {
#define MASTER_END }

#define SLAVE_BEGIN if (op->rank != op->master_rank) {
#define SLAVE_END }

#define X(arr, i, num) (arr)[(i)]
#define Y(arr, i, num) (arr)[(i)+(num)]
#define Z(arr, i, num) (arr)[(i)+(num)*2]
#define VX(arr, i, num) (arr)[(i)+(num)*3]
#define VY(arr, i, num) (arr)[(i)+(num)*4]
#define VZ(arr, i, num) (arr)[(i)+(num)*5]
#define FX(arr, i, num) (arr)[(i)+(num)*6]
#define FY(arr, i, num) (arr)[(i)+(num)*7]
#define FZ(arr, i, num) (arr)[(i)+(num)*8]
#define U(arr, i, num) (arr)[(i)+(num)*9]

#define X_BEGIN(arr, num) (arr)
#define Y_BEGIN(arr, num) ((arr) + (num))
#define Z_BEGIN(arr, num) ((arr) + (num)*2)
#define VX_BEGIN(arr, num) ((arr) + (num)*3)
#define VY_BEGIN(arr, num) ((arr) + (num)*4)
#define VZ_BEGIN(arr, num) ((arr) + (num)*5)
#define FX_BEGIN(arr, num) ((arr) + (num)*6)
#define FY_BEGIN(arr, num) ((arr) + (num)*7)
#define FZ_BEGIN(arr, num) ((arr) + (num)*8)
#define U_BEGIN(arr, num) ((arr) + (num)*9)
#define P_END(arr, num) ((arr) + (num)*10)

#define EPS 1e-3

void compute_fu(double x1, double x2, double y1, double y2, double z1, double z2, double lp, int * bs, double * res) {
    x1 = fmod(x1 + bs[0] * lp * ((int) ((fabs(x1) / ( bs[0] * lp))) + 1), bs[0] * lp);
    y1 = fmod(y1 + bs[1] * lp * ((int) ((fabs(y1) / ( bs[1] * lp))) + 1), bs[1] * lp);
    z1 = fmod(z1 + bs[2] * lp * ((int) ((fabs(z1) / ( bs[2] * lp))) + 1), bs[2] * lp);
    x2 = fmod(x2 + bs[0] * lp * ((int) ((fabs(x2) / ( bs[0] * lp))) + 1), bs[0] * lp);
    y2 = fmod(y2 + bs[1] * lp * ((int) ((fabs(y2) / ( bs[1] * lp))) + 1), bs[1] * lp);
    z2 = fmod(z2 + bs[2] * lp * ((int) ((fabs(z2) / ( bs[2] * lp))) + 1), bs[2] * lp);
    double xx, yy, zz, dist, dist6, force;

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                xx = x2 + lp * (double) (i * bs[0]);
                yy = y2 + lp * (double) (j * bs[1]);
                zz = z2 + lp * (double) (k * bs[2]);
                dist = hypot(hypot(x1 - xx, y1 - yy), z1 - zz);
                if (dist > 2.5) continue;
                if (dist < 0.1) { printf("%e %e %e\n", x1 - xx, y1 - yy, z1 - zz); exit(-1); }
                dist6 = dist * dist * dist * dist * dist * dist;
                res[0] += 4 * (1.0 / dist6 / dist6 - 1.0 / dist6);
                force = 48 * (1.0 / dist6 / dist6 - 0.5 / dist6);
                res[1] += (x1 - xx) * force / dist / dist;
                res[2] += (y1 - yy) * force / dist / dist;
                res[3] += (z1 - zz) * force / dist / dist;
            }
        }
    }
}

void simulate(Options * op) {
    int master = op->master_rank;
    double lp = op->lattice_parameter;
    int * bs = op->box_size;
    int total_particles = op->box_size[0] * op->box_size[1] * op->box_size[2], local_particles = 0;
    double * particles = calloc(10 * total_particles, sizeof(double));
    double * particles2 = calloc(10 * total_particles, sizeof(double));
    double * send_buf = malloc(sizeof(double) * 10 * total_particles);
    double * recv_buf = malloc(sizeof(double) * 60 * total_particles);
    int domens = op->domens[0] * op->domens[1] * op->domens[2];
    int * int_buf = malloc(sizeof(int) * domens);
    int * counts = malloc(sizeof(int) * domens);
    int * displs = malloc(sizeof(int) * domens);
    int * local_indeces = malloc(sizeof(int) * total_particles);
    int * neighbor_indeces[6];
    for (int i = 0; i < 6; i++) neighbor_indeces[i] = malloc(sizeof(int) * total_particles);
    double cell_bounds[6];
    double x, y, z;

    SLAVE_BEGIN
        // set cell bounds
        for (int i = 0; i < 3; i++) {
            cell_bounds[i] = ((double) op->cart_coords[i]) / ((double) op->domens[i]) * lp * bs[i];
            cell_bounds[i+3] = cell_bounds[i] + lp / ((double) op->domens[i]) * bs[i];
        }
    SLAVE_END

    MASTER_BEGIN
        // create particles
        for (int i = 0; i < total_particles; i++) {
            X(particles, i, total_particles) = 0.01 + lp * (i % bs[0]);
            Y(particles, i, total_particles) = 0.01 + lp * ((i / bs[0]) % bs[1]);
            Z(particles, i, total_particles) = 0.01 + lp * ((i / (bs[0] * bs[1])) % bs[2]);
            VX(particles, i, total_particles) = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
            VY(particles, i, total_particles) = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
            VZ(particles, i, total_particles) = op->temperature / 2.0 - ((double) rand()) / ((double) RAND_MAX) * op->temperature;
        }

        memcpy(send_buf, particles, sizeof(double) * 10 * total_particles);
    MASTER_END

    // send initial configuration
    MPI_Bcast(send_buf, 10 * total_particles, MPI_DOUBLE, master, MPI_COMM_WORLD);

    SLAVE_BEGIN
        // get only local particles
        for (int i = 0; i < total_particles; i++) {
            x = X(send_buf, i, total_particles);
            y = Y(send_buf, i, total_particles);
            z = Z(send_buf, i, total_particles);
            x = fmod(x + bs[0] * lp * ((int) ((fabs(x) / ( bs[0] * lp))) + 1), bs[0] * lp);
            y = fmod(y + bs[1] * lp * ((int) ((fabs(y) / ( bs[1] * lp))) + 1), bs[1] * lp);
            z = fmod(z + bs[2] * lp * ((int) ((fabs(z) / ( bs[2] * lp))) + 1), bs[2] * lp);
            if (
                cell_bounds[0] <= x && cell_bounds[3] > x &&
                cell_bounds[1] <= y && cell_bounds[4] > y &&
                cell_bounds[2] <= z && cell_bounds[5] > z
            ) {
                local_indeces[local_particles++] = i;
            }
        }
        for (int i = 0; i < local_particles; i++) {
            X(particles, i, local_particles) = X(send_buf, local_indeces[i], total_particles);
            Y(particles, i, local_particles) = Y(send_buf, local_indeces[i], total_particles);
            Z(particles, i, local_particles) = Z(send_buf, local_indeces[i], total_particles);
            VX(particles, i, local_particles) = VX(send_buf, local_indeces[i], total_particles);
            VY(particles, i, local_particles) = VY(send_buf, local_indeces[i], total_particles);
            VZ(particles, i, local_particles) = VZ(send_buf, local_indeces[i], total_particles);
        }
    SLAVE_END

    for (int step = 0; step < op->total_steps; step++) {
        // perform md step
        SLAVE_BEGIN
            int counts[6], displs[6], to_send;

            // get neighbor local particles sizes
            to_send = local_particles * 3;
            MPI_Neighbor_allgather(&to_send, 1, MPI_INT, counts, 1, MPI_INT, op->comm);
            // init displs
            for (int j = 0; j < 6; j++) displs[j] = j ? displs[j-1] + counts[j-1] : 0;

            // copy local particle coordinates to send buf
            memcpy(send_buf, particles, to_send * sizeof(double));
            // gather neighbor particles
            MPI_Neighbor_allgatherv(send_buf, to_send, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, op->comm);
            for (int i = 0; i < local_particles; i++) {
                // compute forces and energies
                double uf[4] = { 0.0, 0.0, 0.0, 0.0 };
                // consider local particles
                for (int j = 0; j < local_particles; j++) {
                    if (i == j) continue;

                    compute_fu(
                        X(particles, i, local_particles),
                        X(particles, j, local_particles),
                        Y(particles, i, local_particles),
                        Y(particles, j, local_particles),
                        Z(particles, i, local_particles),
                        Z(particles, j, local_particles),
                        lp, bs, uf
                    );
                }


                // consider neighbor particles
                for (int neighbor = 0; neighbor < 6; neighbor++) {
                    if (neighbor < 2 && op->domens[0] == 1) continue;
                    if (neighbor == 1 && op->domens[0] == 2) continue;
                    if (neighbor >= 2 && neighbor < 4 && op->domens[1] == 1) continue;
                    if (neighbor == 3 && op->domens[1] == 2) continue;
                    if (neighbor >= 4 && op->domens[2] == 1) continue;
                    if (neighbor == 5 && op->domens[2] == 2) continue;

                    for (int j = 0; j < counts[neighbor] / 3; j++) {
                        compute_fu(
                            X(particles, i, local_particles),
                            X(recv_buf + displs[neighbor], j, counts[neighbor] / 3),
                            Y(particles, i, local_particles),
                            Y(recv_buf + displs[neighbor], j, counts[neighbor] / 3),
                            Z(particles, i, local_particles),
                            Z(recv_buf + displs[neighbor], j, counts[neighbor] / 3),
                            lp, bs, uf
                        );
                    }
                }

                U(particles, i, local_particles) = uf[0];
                FX(particles, i, local_particles) = uf[1];
                FY(particles, i, local_particles) = uf[2];
                FZ(particles, i, local_particles) = uf[3];
            }

            for (int i = 0; i < local_particles; i++) {
                // change forces and velocities
                VX(particles, i, local_particles) += op->dt * FX(particles, i, local_particles);
                VY(particles, i, local_particles) += op->dt * FY(particles, i, local_particles);
                VZ(particles, i, local_particles) += op->dt * FZ(particles, i, local_particles);
                X(particles, i, local_particles) += op->dt * VX(particles, i, local_particles);
                Y(particles, i, local_particles) += op->dt * VY(particles, i, local_particles);
                Z(particles, i, local_particles) += op->dt * VZ(particles, i, local_particles);
            }

            // distribute particles
            to_send = local_particles * 10;
            MPI_Neighbor_allgather(&to_send, 1, MPI_INT, counts, 1, MPI_INT, op->comm);
            // init displs
            for (int i = 0; i < 6; i++) displs[i] = i ? displs[i-1] + counts[i-1] : 0;
            // copy local particle coordinates to send buf
            memcpy(send_buf, particles, to_send * sizeof(double));
            // gather neighbor particles
            MPI_Neighbor_allgatherv(send_buf, to_send, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, op->comm);

            int new_local_particles = 0, new_neighbor_particles[6] = { 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < local_particles; i++) {
                x = X(particles, i, local_particles);
                y = Y(particles, i, local_particles);
                z = Z(particles, i, local_particles);
                x = fmod(x + bs[0] * lp * ((int) ((fabs(x) / ( bs[0] * lp))) + 1), bs[0] * lp);
                y = fmod(y + bs[1] * lp * ((int) ((fabs(y) / ( bs[1] * lp))) + 1), bs[1] * lp);
                z = fmod(z + bs[2] * lp * ((int) ((fabs(z) / ( bs[2] * lp))) + 1), bs[2] * lp);
                X(particles, i, local_particles) = x;
                Y(particles, i, local_particles) = y;
                Z(particles, i, local_particles) = z;
                if (
                    cell_bounds[0] <= x && cell_bounds[3] > x &&
                    cell_bounds[1] <= y && cell_bounds[4] > y &&
                    cell_bounds[2] <= z && cell_bounds[5] > z
                ) local_indeces[new_local_particles++] = i;
            }

            for (int neighbor = 0; neighbor < 6; neighbor++) {
                if (neighbor < 2 && op->domens[0] == 1) continue;
                if (neighbor == 1 && op->domens[0] == 2) continue;
                if (neighbor >= 2 && neighbor < 4 && op->domens[1] == 1) continue;
                if (neighbor == 3 && op->domens[1] == 2) continue;
                if (neighbor >= 4 && op->domens[2] == 1) continue;
                if (neighbor == 5 && op->domens[2] == 2) continue;

                for (int j = 0; j < counts[neighbor] / 10; j++) {
                    x = X(recv_buf + displs[neighbor], j, counts[neighbor] / 10);
                    y = Y(recv_buf + displs[neighbor], j, counts[neighbor] / 10);
                    z = Z(recv_buf + displs[neighbor], j, counts[neighbor] / 10);
                    x = fmod(x + bs[0] * lp * ((int) ((fabs(x) / ( bs[0] * lp))) + 1), bs[0] * lp);
                    y = fmod(y + bs[1] * lp * ((int) ((fabs(y) / ( bs[1] * lp))) + 1), bs[1] * lp);
                    z = fmod(z + bs[2] * lp * ((int) ((fabs(z) / ( bs[2] * lp))) + 1), bs[2] * lp);

                    if (
                        cell_bounds[0] <= x && cell_bounds[3] > x &&
                        cell_bounds[1] <= y && cell_bounds[4] > y &&
                        cell_bounds[2] <= z && cell_bounds[5] > z
                    ) neighbor_indeces[neighbor][new_neighbor_particles[neighbor]++] = j;
                }
            }
            memcpy(particles2, particles, sizeof(double) * to_send);
            int old_local_particles = local_particles;
            local_particles = new_local_particles;
            for (int i = 0; i < 6; i++) local_particles += new_neighbor_particles[i];
            for (int i = 0; i < new_local_particles; i++) {
                X(particles, i, local_particles) = X(particles2, local_indeces[i], old_local_particles);
                Y(particles, i, local_particles) = Y(particles2, local_indeces[i], old_local_particles);
                Z(particles, i, local_particles) = Z(particles2, local_indeces[i], old_local_particles);
                VX(particles, i, local_particles) = VX(particles2, local_indeces[i], old_local_particles);
                VY(particles, i, local_particles) = VY(particles2, local_indeces[i], old_local_particles);
                VZ(particles, i, local_particles) = VZ(particles2, local_indeces[i], old_local_particles);
                FX(particles, i, local_particles) = FX(particles2, local_indeces[i], old_local_particles);
                FY(particles, i, local_particles) = FY(particles2, local_indeces[i], old_local_particles);
                FZ(particles, i, local_particles) = FZ(particles2, local_indeces[i], old_local_particles);
                U(particles, i, local_particles) = U(particles2, local_indeces[i], old_local_particles);
            }
            int k = new_local_particles;
            for (int neighbor = 0; neighbor < 6; neighbor++) {
                for (int i = 0; i < new_neighbor_particles[neighbor]; i++) {
                    X(particles, k, local_particles) = X(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    Y(particles, k, local_particles) = Y(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    Z(particles, k, local_particles) = Z(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    VX(particles, k, local_particles) = VX(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    VY(particles, k, local_particles) = VY(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    VZ(particles, k, local_particles) = VZ(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    FX(particles, k, local_particles) = FX(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    FY(particles, k, local_particles) = FY(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    FZ(particles, k, local_particles) = FZ(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    U(particles, k, local_particles) = U(recv_buf + displs[neighbor], neighbor_indeces[neighbor][i], counts[neighbor] / 10);
                    k++;
                }
            }
        SLAVE_END

        if (step % op->dump_every == 0) {
            // dump and print energy
            int to_send = 0;
            SLAVE_BEGIN
                to_send = local_particles * 7;
            SLAVE_END

            MPI_Gather(&to_send, 1, MPI_INT, int_buf, 1, MPI_INT, op->master_rank, MPI_COMM_WORLD);

            MASTER_BEGIN
                for (int i = 0; i < op->domens[0] * op->domens[1] * op->domens[2]; i++) {
                    counts[i] = int_buf[i];
                    displs[i] = i ? displs[i-1] + counts[i] : 0;
                }
            MASTER_END

            SLAVE_BEGIN
                memcpy(send_buf, particles, 6 * local_particles * sizeof(double));
                memcpy(send_buf + local_particles * 6, U_BEGIN(particles, local_particles), local_particles * sizeof(double));
            SLAVE_END

            MPI_Gatherv(send_buf, to_send, MPI_DOUBLE, recv_buf, counts, displs, MPI_DOUBLE, op->master_rank, MPI_COMM_WORLD);

            MASTER_BEGIN
                if (step == 0) printf("step,T,U,E\n");
                double T = 0.0, u = 0.0;
                char strbuf[4096];
                sprintf(strbuf, "%s/%d.atom", op->output_dir, step);
                FILE * out = fopen(strbuf, "w");
                fprintf(out, "ITEM: TIMESTEP\n%d\n", step);
                fprintf(out, "ITEM: NUMBER OF ATOMS\n%d\n", total_particles);
                fprintf(out, "ITEM: BOX BOUNDS pp pp pp\n0.0 %lf\n0.0 %lf\n0.0 %lf\n", bs[0] * lp, bs[1] * lp, bs[2] * lp);
                fprintf(out, "ITEM: ATOMS x y z vx vy vz\n");
                int count = 0;
                for (int j = 0; j < domens; j++) {
                    for (int k = 0; k < counts[j] / 7; k++) {
                        fprintf(
                            out, "%lf %lf %lf %lf %lf %lf\n",
                            X(recv_buf + displs[j], k, counts[j] / 7),
                            Y(recv_buf + displs[j], k, counts[j] / 7),
                            Z(recv_buf + displs[j], k, counts[j] / 7),
                            VX(recv_buf + displs[j], k, counts[j] / 7),
                            VY(recv_buf + displs[j], k, counts[j] / 7),
                            VZ(recv_buf + displs[j], k, counts[j] / 7)
                        );
                        T += 0.5 * hypot(hypot(VX(recv_buf + displs[j], k, counts[j] / 7), VY(recv_buf + displs[j], k, counts[j] / 7)), VZ(recv_buf + displs[j], k, counts[j] / 7));
                        u += FX(recv_buf + displs[j], k, counts[j] / 7);
                        count++;
                    }
                }

                fclose(out);

                printf("%d,%lf,%lf,%lf\n", step, T, u, T+u);
            MASTER_END
        }
    }

    free(int_buf);
    free(counts);
    free(displs);
    free(local_indeces);
    for (int i = 0; i < 6; i++) free(neighbor_indeces[i]);
    free(recv_buf);
    free(send_buf);
    free(particles);
    free(particles2);
}

int main(int argc, char * argv[]) {
    Options * options = parse_cla(argc, argv);
    if (!options) goto error;

    struct stat st = {0};
    if (stat(options->output_dir, &st) == -1 && !mkdir(options->output_dir, 0777)) goto error;

    int rank = init_mpi(argc, argv, options);
    if (-1 == rank) goto error;

    simulate(options);

    options_destroy(options);
    finalize_mpi();

    return 0;

error:
    return 1;
}
