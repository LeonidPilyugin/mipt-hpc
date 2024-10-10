#pragma once

#ifdef __cplusplus
#error "Compile using C compiler, not C++"
#endif

typedef float f_t;
typedef int i_t;

#define f_s sizeof(f_t)
#define i_s sizeof(i_t)

typedef f_t * f_a;

