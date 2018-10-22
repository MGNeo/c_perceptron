#ifndef C_PERCEPTRON_H
#define C_PERCEPTRON_H

#include <stddef.h>
#include <stdint.h>

typedef struct s_c_perceptron c_perceptron;

typedef struct s_c_pgs c_pgs;

c_perceptron *c_perceptron_create(const size_t _layers_count,
                                  const size_t *const _topology,
                                  size_t *const _error);

ptrdiff_t c_perceptron_delete(c_perceptron *const _perceptron);

ptrdiff_t c_perceptron_noise(c_perceptron *const _perceptron,
                             const float _noise_force,
                             uint64_t *const _seed);

float *c_perceptron_get_ins(c_perceptron *const _perceptron);

const float *c_perceptron_get_outs(c_perceptron *const _perceptron);

ptrdiff_t c_perceptron_execute(c_perceptron *const _perceptron);

c_perceptron *c_perceptron_clone(const c_perceptron *const _perceptron,
                                 size_t *const _error);

ptrdiff_t c_perceptron_save(const c_perceptron *const _perceptron,
                            const char *const _file_name);

c_perceptron *c_perceptron_load(const char *const _file_name,
                                size_t *const _error);

// --------------------

c_pgs *c_pgs_create(const c_perceptron *const _perceptron,
                    const size_t _pop_count,
                    size_t *const _error);

ptrdiff_t c_pgs_delete(c_pgs *const _pgs);

ptrdiff_t c_pgs_run(c_pgs *const _pgs,
                    c_perceptron *const _perceptron,
                    const float *const _lessons,
                    const size_t _lessons_count,
                    const size_t _iterations_count,
                    const float _noise_force,
                    const float _mut_force,
                    uint64_t *const _seed);

#endif
