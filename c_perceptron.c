#include "c_perceptron.h"

#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>

#define A 6364136223846793005LLU
#define C 1
#define RAND_64_32_MAX UINT32_MAX

// Перцептрон.
struct s_c_perceptron
{
    size_t layers_count;
    size_t *topology;

    size_t weights_count;
    float *weights;

    float *ins;
    float *outs;
};

// Сущность с весами и ошибкой.
// Необходима для оптимизации производительности и расхода памяти при работе генетического алгоритма.
typedef struct s_c_weights_and_sigma
{
    float *weights;
    float sigma;// Суммарная ошибка по всем сигналам всех уроков.
} c_weights_and_sigma;

// Перцептронный генетический селекционер.
struct s_c_pgs
{
    size_t layers_count;
    size_t *topology;

    size_t pop_count;
    c_weights_and_sigma *pop;

    size_t pool_count;
    c_weights_and_sigma *pool;
};

// Потоконезависимый ГПСЧ с периодом 2^64 и диапазоном генерируемых значений [0; UINT32_MAX].
static uint32_t rand_64_32(uint64_t *const _seed)
{
    if (_seed == NULL) return 0;

    *_seed = *_seed * A + C;

    return *_seed >> 32;
}

// Функция скрещивания и мутирования весов.
static void weights_cross_and_mut(const float *const _weights_1,
                                  const float *const _weights_2,
                                  float *const _weights_3,
                                  const size_t _weights_count,
                                  const float _mut_force,
                                  uint64_t *const _seed)
{
    if ( (_weights_1 == NULL) ||
         (_weights_2 == NULL) ||
         (_weights_3 == NULL) ||
         (_weights_count == 0) )
    {
        return;
    }

    for (size_t w = 0; w < _weights_count; ++w)
    {
        // Наследуем вес с равной вероятностью от одного из предков.
        if (rand_64_32(_seed) % 2 == 0)
        {
            _weights_3[w] = _weights_1[w];
        } else {
            _weights_3[w] = _weights_2[w];
        }

        // Вероятность мутации веса при наследовании 5%.
        if (rand_64_32(_seed) % 20 == 0)
        {
            const float sign = pow(-1, rand_64_32(_seed) % 2);
            const float value = rand_64_32(_seed) / (float)(RAND_64_32_MAX);
            _weights_3[w] += sign * value * _mut_force;
        }
    }
}

// Обменивает местами значения указателей на float.
// Даже если заданные указатели хранят NULL.
static void float_ptr_swap(float **const _weights_1,
                           float **const _weights_2)
{
    if ( (_weights_1 == NULL) ||
         (_weights_2 == NULL) )
    {
        return;
    }

    float *const h = *_weights_1;
    *_weights_1 = *_weights_2;
    *_weights_2 = h;
}

// Заполняет заданные веса шумом.
static void weights_noise(float *const _weights,
                          const size_t _weights_count,
                          const float _noise_force,
                          uint64_t *const _seed)
{
    if ( (_weights == NULL) ||
         (_weights_count == 0) )
    {
        return;
    }

    for (size_t w = 0; w < _weights_count; ++w)
    {
        const float sign = pow(-1, rand_64_32(_seed) % 2);
        const float value = rand_64_32(_seed) / (float) (RAND_64_32_MAX);
        _weights[w] = sign * value * _noise_force;
    }
}

// Компаратор, сортирует по возрастанию ошибки сущности.
static int comp(const void *_p1,
                const void *_p2)
{
    const c_weights_and_sigma *const p1 = _p1;
    const c_weights_and_sigma *const p2 = _p2;
    if (p1->sigma > p2->sigma)
    {
        return 1;
    } else {
        if (p1->sigma == p2->sigma)
        {
            return 0;
        } else {
            return -1;
        }
    }
}

// Функция активации.
static float activation_function(const float _value)
{
    return 1 / (1 + exp(-_value));// Возможно, функция, имеющая значения [-1; +1] будет лучше?
}

// Если расположение задано, в него помещается код.
static void error_set(size_t *const _error,
                      const size_t _code)
{
    if (_error != NULL)
    {
        *_error = _code;
    }
}

// Создает перцептрон заданой топологии.
// Слоев должно быть >= 2..
// Каждый слой должен содержать > 0 нейронов.
// В случае ошибки возвращает NULL, и если _error != NULL,
// в заданное расположение помещается код причины ошибки (> 0).
c_perceptron *c_perceptron_create(const size_t _layers_count,
                                  const size_t *const _topology,
                                  size_t *const _error)
{
    if (_layers_count < 2)
    {
        error_set(_error, 1);
        return NULL;
    }
    if (_topology == NULL)
    {
        error_set(_error, 2);
        return NULL;
    }
    for (size_t l = 0; l < _layers_count; ++l)
    {
        if (_topology[l] == 0)
        {
            error_set(_error, 3);
            return NULL;
        }
    }

    // Определим, сколько памяти нужно под топологию.
    const size_t new_topology_size = sizeof(size_t) * _layers_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_topology_size == 0) ||
         (new_topology_size / sizeof(size_t) != _layers_count) )
    {
        error_set(_error, 4);
        return NULL;
    }

    // Попытаемся выделить память под топологию.
    size_t *const new_topology = malloc(new_topology_size);

    // Контроль успешности выделения памяти.
    if (new_topology == NULL)
    {
        error_set(_error, 5);
        return NULL;
    }

    // Определим количество весов в перцептроне.
    size_t new_weights_count = 0;
    for (size_t l = 1; l < _layers_count; ++l)
    {
        const size_t m = _topology[l - 1] * _topology[l];
        // Контроль целочисленного переполнения при умножении.
        if ( (m == 0) ||
             (m / _topology[l - 1] != _topology[l]) )
        {
            free(new_topology);
            error_set(_error, 6);
            return NULL;
        }
        const size_t s = new_weights_count + m;
        // Контроль целочисленного переполнения при сложении.
        if (s < new_weights_count)
        {
            free(new_topology);
            error_set(_error, 7);
            return NULL;
        }
        new_weights_count = s;
    }

    // Определим, сколько памяти нужно под веса.
    const size_t new_weights_size = sizeof(float) * new_weights_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_weights_size == 0) ||
         (new_weights_size / sizeof(float) != new_weights_count) )
    {
        free(new_topology);
        error_set(_error, 8);
        return NULL;
    }

    // Попытаемся выделить память под веса.
    float *const new_weights = malloc(new_weights_size);
    // Контроль успешности выделения памяти.
    if (new_weights == NULL)
    {
        free(new_topology);
        error_set(_error, 9);
        return NULL;
    }

    // Определяем, сколько памяти нужно под входа.
    const size_t new_ins_count = _topology[0];
    const size_t new_ins_size = sizeof(float) * new_ins_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_ins_size == 0) ||
         (new_ins_size / sizeof(float) != new_ins_count) )
    {
        free(new_weights);
        free(new_topology);
        error_set(_error, 10);
        return NULL;
    }

    // Пытаемся выделить память под входа.
    float *const new_ins = malloc(new_ins_size);
    // Контроль успешности выделения памяти.
    if (new_ins == NULL)
    {
        free(new_weights);
        free(new_topology);
        error_set(_error, 11);
        return NULL;
    }

    // Определяем, сколько памяти нужно под выхода.
    const size_t new_outs_count = _topology[_layers_count - 1];
    const size_t new_outs_size = sizeof(float) * new_outs_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_outs_size == 0) ||
         (new_outs_size / sizeof(float) != new_outs_count) )
    {
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 12);
        return NULL;
    }

    // Попытаемся выделить память под выхода.
    float *new_outs = malloc(new_outs_size);
    // Контроль успешности выделения памяти.
    if (new_outs == NULL)
    {
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 13);
        return NULL;
    }

    // Пытаемся выделить память под перцептрон.
    c_perceptron *const new_perceptron = malloc(sizeof(c_perceptron));

    // Контроль успешности выделения памяти.
    if (new_perceptron == NULL)
    {
        free(new_outs);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 14);
        return NULL;
    }

    // Собираем перцептрон.
    new_perceptron->layers_count = _layers_count;
    new_perceptron->topology = new_topology;
    memcpy(new_topology, _topology, new_topology_size);
    new_perceptron->weights_count = new_weights_count;
    new_perceptron->weights = new_weights;
    new_perceptron->ins = new_ins;
    new_perceptron->outs = new_outs;

    return new_perceptron;
}

// Удаляет перцептрон.
// В случае успеха возвращает > 0.
// В случае ошибки возвращает < 0.
ptrdiff_t c_perceptron_delete(c_perceptron *const _perceptron)
{
    if (_perceptron == NULL)
    {
        return -1;
    }

    free(_perceptron->outs);
    free(_perceptron->ins);
    free(_perceptron->weights);
    free(_perceptron->topology);
    free(_perceptron);

    return 1;
}

// Заполняет веса перцептрона шумом.
// В случае успеха функция возвращает > 0.
// В случае ошибки функция возвращает < 0.
ptrdiff_t c_perceptron_noise(c_perceptron *const _perceptron,
                             const float _noise_force,
                             uint64_t *const _seed)
{
    if (_perceptron == NULL)
    {
        return -1;
    }

    if (_seed == NULL)
    {
        return -2;
    }

    weights_noise(_perceptron->weights, _perceptron->weights_count, _noise_force, _seed);

    return 1;
}


// Прямое обращение ко входным сигналам перцептрона.
// В случае, если _perceptron == NULL, возвращает NULL.
float *c_perceptron_get_ins(c_perceptron *const _perceptron)
{
    if (_perceptron == NULL)
    {
        return NULL;
    }

    return _perceptron->ins;
}

// Прямое обращение к выходным сигналам перцептрона.
// В случае, если _perceptron == NULL, возвращает NULL.
const float *c_perceptron_get_outs(c_perceptron *const _perceptron)
{
    if (_perceptron == NULL)
    {
        return NULL;
    }

    return _perceptron->outs;
}

// Пропускает сигнал через перцептрон.
// В случае успеха возвращает > 0.
// В случае ошибки возвращает < 0.
ptrdiff_t c_perceptron_execute(c_perceptron *const _perceptron)
{
    if (_perceptron == NULL)
    {
        return -1;
    }

    // Определяем, сколько нейронов имеется в самом "жирном" слое.
    size_t h_buffer_count = 0;
    for (size_t l = 0; l < _perceptron->layers_count; ++l)
    {
        if (_perceptron->topology[l] > h_buffer_count)
        {
            h_buffer_count = _perceptron->topology[l];
        }
    }

    // Вспомогательные буфера.
    float a[h_buffer_count],
          b[h_buffer_count];

    // Указатели нужны для быстрого свопа.
    float *h_ins = a,
          *h_outs = b;

    // Определяем, сколько памяти занимают входные сигналы перцептрона.
    const size_t ins_size = sizeof(float) * _perceptron->topology[0];
    // Контроль целочисленного переполнения не нужен, так как
    // он выполняется на этапе конструирования перцептрона.

    // Определяем, сколько памяти занимают выходные сигналы перцептрона.
    const size_t outs_size = sizeof(float) * _perceptron->topology[_perceptron->layers_count - 1];
    // Контроль целочисленного переполнения не нужен, так как
    // он выполняется на этапе конструирования перцептрона.

    // Пропускаем входные сигналы перцептрона через сеть.

    memcpy(h_outs, _perceptron->ins, ins_size);
    size_t w = 0;
    for (size_t l = 1; l < _perceptron->layers_count; ++l)
    {
        float *const h = h_ins;
        h_ins = h_outs;
        h_outs = h;

        for (size_t cn = 0; cn < _perceptron->topology[l]; ++cn)
        {
            float sum = 0;
            for (size_t pn = 0; pn < _perceptron->topology[l - 1]; ++pn)
            {
                sum += h_ins[pn] * _perceptron->weights[w++];
            }
            h_outs[cn] = activation_function(sum);
        }
    }

    // Помещаем итоговые сигналы на выход перцептрона.
    memcpy(_perceptron->outs, h_outs, outs_size);

    return 1;
}

// Клонирует перцептрон.
// В случае ошибки возвращает NULL, и если _error != NULL, в заданное расположение
// помещается код причины ошибки (> 0).
c_perceptron *c_perceptron_clone(const c_perceptron *const _perceptron,
                                 size_t *const _error)
{
    if (_perceptron == NULL)
    {
        error_set(_error, 1);
        return NULL;
    }

    // Определим, сколько памяти необходимо под тополгию.
    const size_t new_topology_size = sizeof(size_t) * _perceptron->layers_count;
    // Контроль целочисленного переполнения не нужен, так как
    // это переполнение првоеряется на этапе конструирования перцептрона.

    // Попытаемся выделить память под топологию.
    size_t *const new_topology = malloc(new_topology_size);

    // Проверка успешности выделения памяти.
    if (new_topology == NULL)
    {
        error_set(_error, 2);
        return NULL;
    }

    // Определим, сколько памяти необходимо под веса.
    const size_t new_weights_size = sizeof(float) * _perceptron->weights_count;
    // Контроль целочисленного переполнения не нужен, так как
    // это переполнение контролируется на этапе конструирования перцептрона.

    // Попытаемся выделить память под веса.
    float *const new_weights = malloc(new_weights_size);

    // Проверка успешности выделения памяти.
    if (new_weights == NULL)
    {
        free(new_topology);
        error_set(_error, 3);
        return NULL;
    }

    // Определим, сколько памяти необходимо под входа.
    const size_t new_ins_size = sizeof(float) * _perceptron->topology[0];
    // Контроль целочисленного переполнения не нужен, так как
    // это переполнение контролируется на этапе конструирования перцептрона.

    // Попытаемся выделить память под входа.
    float *const new_ins = malloc(new_ins_size);

    // Проверка успешности выделения памяти.
    if (new_ins == NULL)
    {
        free(new_weights);
        free(new_topology);
        error_set(_error, 4);
        return NULL;
    }

    // Определим, сколько памяти необходимо под выхода.
    const size_t new_outs_size = sizeof(float) * _perceptron->topology[_perceptron->layers_count - 1];
    // Контроль целочисленного переполнения не нужен, так как
    // это переполнение контролируется на этапе конструирования перцептрона.

    // Попытаемся выделить память под выхода.
    float *const new_outs = malloc(new_outs_size);

    // Проверка успешности выделения памяти.
    if (new_outs == NULL)
    {
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 5);
        return NULL;
    }

    // Попытаемся выделить память под перцептрон.
    c_perceptron *const new_perceptron = malloc(sizeof(c_perceptron));

    // Контроль успешности выделения памяти.
    if (new_perceptron == NULL)
    {
        free(new_outs);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 6);
        return NULL;
    }

    // Собираем перцептрон.
    new_perceptron->layers_count = _perceptron->layers_count;
    new_perceptron->topology = new_topology;
    memcpy(new_topology, _perceptron->topology, new_topology_size);
    new_perceptron->weights_count = _perceptron->weights_count;
    new_perceptron->weights = new_weights;
    memcpy(new_weights, _perceptron->weights, new_weights_size);
    new_perceptron->ins = new_ins;
    memcpy(new_ins, _perceptron->ins, new_ins_size);
    new_perceptron->outs = new_outs;
    memcpy(new_outs, _perceptron->outs, new_outs_size);

    return new_perceptron;
}

// Сохраняет перцептрон в двоичный файл в платформозависимом формате (порядок байт и размер size_t платформозависимы).
// Если файл с заданным именем существует, то он перезаписывается, если это возможно (если невозможно, функция вернет < 0).
// В случае успеха возвращает > 0.
// В случае ошибки возвращает < 0.
ptrdiff_t c_perceptron_save(const c_perceptron *const _perceptron,
                            const char *const _file_name)
{
    if (_perceptron == NULL)
    {
        return -1;
    }
    if (_file_name == NULL)
    {
        return -2;
    }
    if (strlen(_file_name) == 0)
    {
        return -3;
    }

    // Вещи, которые зависят от платформы:
    // 1 - размер size_t;
    // 2 - порядок size_t;
    // 3 - порядок float.

    FILE *f = fopen(_file_name, "wb");

    // Контроль успешности открытия.
    if (f == NULL)
    {
        return -4;
    }

    int r_code;
    // Записываем в файл количество слоев.
    r_code = fwrite(&_perceptron->layers_count, sizeof(size_t), 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -5;
    }
    // Записываем в файл топологию.
    r_code = fwrite(_perceptron->topology, sizeof(size_t) * _perceptron->layers_count, 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -6;
    }

    // Записываем в файл количество весов.
    r_code = fwrite(&_perceptron->weights_count, sizeof(size_t), 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -7;
    }

    // Записываем в файл веса.
    r_code = fwrite(_perceptron->weights, sizeof(float) * _perceptron->weights_count, 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -8;
    }

    // Записываем в файл входные сигналы.
    r_code = fwrite(_perceptron->ins, sizeof(float)*_perceptron->topology[0], 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -9;
    }

    // Записываем в файл выходные сигналы.
    r_code = fwrite(_perceptron->outs, sizeof(float) * _perceptron->topology[_perceptron->layers_count - 1], 1, f);

    // Контроль успешности записи.
    if (r_code != 1)
    {
        fclose(f);
        return -10;
    }

    fclose(f);

    return 1;
}

// Загружает перцептрон из файла в платформозависимом формате (порядок байт и размер size_t платформозависимы).
// В случае ошибки возвращает NULL, и если _error != NULL, в заданное расположение
// помещается код причины ошибки (> 0).
c_perceptron *c_perceptron_load(const char *const _file_name,
                                size_t *const _error)
{
    if (_file_name == NULL)
    {
        error_set(_error, 1);
        return NULL;
    }
    if (strlen(_file_name) == 0)
    {
        error_set(_error, 2);
        return NULL;
    }

    FILE *f = fopen(_file_name, "rb");

    // Контроль успешности открытия.
    if (f == NULL)
    {
        error_set(_error, 3);
        return NULL;
    }

    int r_code;

    // Считываем количество слоев.
    size_t new_layers_count;
    r_code = fread(&new_layers_count, sizeof(size_t), 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        error_set(_error, 4);
        return NULL;
    }

    // Проверяем число слоев на валидность.
    if (new_layers_count < 2)
    {
        fclose(f);
        error_set(_error, 5);
        return NULL;
    }

    // Определяем, сколько памяти нужно под топологию.
    const size_t new_topology_size = sizeof(size_t) * new_layers_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_topology_size == 0) ||
         (new_topology_size / sizeof(size_t) != new_layers_count) )
    {
        fclose(f);
        error_set(_error, 6);
        return NULL;
    }

    // Пытаемся выделить память под топологию.
    size_t *const new_topology = malloc(new_topology_size);

    // Контроль успешности выделения памяти.
    if (new_topology == NULL)
    {
        fclose(f);
        error_set(_error, 7);
        return NULL;
    }

    // Попытаемся считать из файла топологию.
    r_code = fread(new_topology, new_topology_size, 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        free(new_topology);
        error_set(_error, 8);
        return NULL;
    }

    // Контроль корректности топологии.
    for (size_t l = 0; l < new_layers_count; ++l)
    {
        if (new_topology[l] == 0)
        {
            fclose(f);
            free(new_topology);
            error_set(_error, 9);
            return NULL;
        }
    }

    // Считываем количество весов.
    size_t new_weights_count;
    r_code = fread(&new_weights_count, sizeof(size_t), 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        free(new_topology);
        error_set(_error, 10);
        return NULL;
    }

    // Определяем, сколько весов должно быть у загруженной топологии.
    size_t h_weights_count = 0;
    for (size_t l = 1; l < new_layers_count; ++l)
    {
        const size_t m = new_topology[l - 1] * new_topology[l];
        // Контроль целочисленного переполнения при умножении.
        if ( (m == 0) ||
             (m / new_topology[l - 1] != new_topology[l]) )
        {
            fclose(f);
            free(new_topology);
            error_set(_error, 11);
            return NULL;
        }
        const size_t s = h_weights_count + m;
        // Контроль целочисленного переполнения при сложении.
        if (s < h_weights_count)
        {
            fclose(f);
            free(new_topology);
            error_set(_error, 11);
            return NULL;
        }
        h_weights_count += m;
    }

    // Проверяем, совпадает ли это число, с загруженным из файла.
    if (h_weights_count != new_weights_count)
    {
        fclose(f);
        free(new_topology);
        error_set(_error, 12);
        return NULL;
    }

    // Определим размер весов, которые необходимо загрузить из файла.
    const size_t new_weights_size = sizeof(float) * new_weights_count;

    // Контроль целочисленного переполнения при умножении.
    if (new_weights_size < sizeof(float))
    {
        fclose(f);
        free(new_topology);
        error_set(_error, 13);
        return NULL;
    }

    // Попытаемся выделить память под веса.
    float *const new_weights = malloc(new_weights_size);

    // Контроль успешности выделения памяти.
    if (new_weights == NULL)
    {
        fclose(f);
        free(new_topology);
        error_set(_error, 14);
        return NULL;
    }

    // Попытаемся считать веса из файла.
    r_code = fread(new_weights, new_weights_size, 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        free(new_weights);
        free(new_topology);
        error_set(_error, 15);
        return NULL;
    }

    // Определим, сколько памяти нужно под входа перцептрона.
    const size_t new_ins_size = sizeof(float) * new_topology[0];

    // Контроль целочисленного переполнения при сложении.
    if (new_ins_size < sizeof(float))
    {
        fclose(f);
        free(new_weights);
        free(new_topology);
        error_set(_error, 16);
        return NULL;
    }

    // Попытаемся выделить память под входа перцептрона.
    float *const new_ins = malloc(new_ins_size);

    // Контроль успешности выделения памяти.
    if (new_ins == NULL)
    {
        fclose(f);
        free(new_weights);
        free(new_topology);
        error_set(_error, 17);
        return NULL;
    }

    // Попытаемся считать входа.
    r_code = fread(new_ins, new_ins_size, 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        free(new_weights);
        free(new_topology);
        error_set(_error, 18);
        return NULL;
    }

    // Определим, сколько памяти нужно под выхода перцептрона.
    const size_t new_outs_size = sizeof(float) * new_topology[new_layers_count - 1];

    // Контроль целочисленного переполнения при сложении.
    if ( (new_outs_size == 0) ||
         (new_outs_size / sizeof(float) != new_topology[new_layers_count - 1]) )
    {
        fclose(f);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 19);
        return NULL;
    }

    // Попытаемся выделить память под выхода.
    float *const new_outs = malloc(new_outs_size);

    // Контроль успешности выделения памяти.
    if (new_outs == NULL)
    {
        fclose(f);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 20);
        return NULL;
    }

    // Попытаемся считать выхода.
    r_code = fread(new_outs, new_outs_size, 1, f);

    // Контроль успешности считывания.
    if (r_code != 1)
    {
        fclose(f);
        free(new_outs);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 21);
        return NULL;
    }

    fclose(f);

    // Попытаемся выделить память под перцептрон.
    c_perceptron *const new_perceptron = malloc(sizeof(c_perceptron));

    // Контроль успешности выделения памяти.
    if (new_perceptron == NULL)
    {
        free(new_outs);
        free(new_ins);
        free(new_weights);
        free(new_topology);
        error_set(_error, 22);
        return NULL;
    }

    // Собираем перцептрон.
    new_perceptron->layers_count = new_layers_count;
    new_perceptron->topology = new_topology;
    new_perceptron->weights_count = new_weights_count;
    new_perceptron->weights = new_weights;
    new_perceptron->ins = new_ins;
    new_perceptron->outs = new_outs;

    return new_perceptron;
}

// Создает перцептронного генетического селекционера.
// Популяция должна быть >= 10.
// В случае ошибки возвращает NULL, и если _error != NULL,
// в заданное расположение помещается код причины ошибки (> 0).
c_pgs *c_pgs_create(const c_perceptron *const _perceptron,
                    const size_t _pop_count,
                    size_t *const _error)
{
    if (_perceptron == NULL)
    {
        error_set(_error, 1);
        return NULL;
    }
    if (_pop_count < 10)
    {
        error_set(_error, 2);
        return NULL;
    }

    // Определим, сколько памяти необходимо под топологию.
    const size_t new_topology_size = sizeof(size_t) * _perceptron->layers_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_topology_size == 0) ||
         (new_topology_size / sizeof(size_t) != _perceptron->layers_count) )
    {
        error_set(_error, 3);
        return NULL;
    }

    // Попытаемся выделить память под топологию.
    size_t *const new_topology = malloc(new_topology_size);
    // Контроль успешности выделения памяти.
    if (new_topology == NULL)
    {
        error_set(_error, 4);
        return NULL;
    }

    // Определим, сколько памяти необходимо под популяцию.
    const size_t new_pop_size = sizeof(c_weights_and_sigma) * _pop_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_pop_size == 0) ||
         (new_pop_size / sizeof(c_weights_and_sigma) != _pop_count) )
    {
        free(new_topology);
        error_set(_error, 5);
        return NULL;
    }

    // Попытаемся выделить память под популяцию.
    c_weights_and_sigma *const new_pop = malloc(new_pop_size);
    // Контроль успешности выделения памяти.
    if (new_pop == NULL)
    {
        free(new_topology);
        error_set(_error, 6);
        return NULL;
    }

    // Определим количество мест в пуле.
    size_t new_pool_count = _pop_count * _pop_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_pool_count == 0) ||
         (new_pool_count / _pop_count != _pop_count) )
    {
        free(new_pop);
        free(new_topology);
        error_set(_error, 7);
        return NULL;
    }
    new_pool_count -= _pop_count;

    // Определим, сколько памяти необходимо под пул.
    const size_t new_pool_size = sizeof(c_weights_and_sigma) * new_pool_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_pool_size == 0) ||
         (new_pool_size / sizeof(c_weights_and_sigma) != new_pool_count) )
    {
        free(new_pop);
        free(new_topology);
        error_set(_error, 8);
        return NULL;
    }

    // Попытаемся выделить память под пул.
    c_weights_and_sigma *const new_pool = malloc(new_pool_size);
    // Контроль успешности выделения памяти.
    if (new_pool == NULL)
    {
        free(new_pop);
        free(new_topology);
        error_set(_error, 9);
        return NULL;
    }

    // Определим, сколько памяти занимают все веса перцептрона.
    const size_t new_weights_size = sizeof(float) * _perceptron->weights_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (new_weights_size == 0) ||
         (new_weights_size / sizeof(float) != _perceptron->weights_count) )
    {
        free(new_pool);
        free(new_pop);
        free(new_topology);
        error_set(_error, 10);
        return NULL;
    }

    // Обеспечиваем весами каждую сущность популяции.
    for (size_t p = 0; p < _pop_count; ++p)
    {
        // Пытаемся выделить память под веса.
        float *const new_weights = malloc(new_weights_size);
        // Контроль успешности выделения памяти.
        if (new_weights == NULL)
        {
            for (size_t d = 0; d < p; ++d)
            {
                free(new_pop[d].weights);
            }
            free(new_pool);
            free(new_pop);
            free(new_topology);
            error_set(_error, 11);
            return NULL;
        }
        new_pop[p].weights = new_weights;
    }

    // Обеспечиваем весами каждую сущность пула.
    for (size_t p = 0; p < new_pool_count; ++p)
    {
        // Пытаемся выделить память под веса.
        float *const new_weights = malloc(new_weights_size);
        // Контроль успещности выделения памяти.
        if (new_weights == NULL)
        {
            for (size_t d = 0; d < p; ++d)
            {
                free(new_pool[d].weights);
            }
            for (size_t d = 0; d < _pop_count; ++d)
            {
                free(new_pop[d].weights);
            }
            free(new_pool);
            free(new_pop);
            free(new_topology);
            error_set(_error, 12);
            return NULL;
        }
        new_pool[p].weights = new_weights;
    }

    // Пытаемся выделить память под c_pgs.
    c_pgs *const new_pgs = malloc(sizeof(c_pgs));
    // Контроль успешности выделения памяти.
    if (new_pgs == NULL)
    {
        for (size_t d = 0; d < new_pool_count; ++d)
        {
            free(new_pool[d].weights);
        }
        for (size_t d = 0; d < _pop_count; ++d)
        {
            free(new_pop[d].weights);
        }
        free(new_pool);
        free(new_pop);
        free(new_topology);
        error_set(_error, 13);
        return NULL;
    }

    // Собираем c_pgs.
    new_pgs->layers_count = _perceptron->layers_count;
    new_pgs->topology = new_topology;
    memcpy(new_topology, _perceptron->topology, new_topology_size);
    new_pgs->pop_count = _pop_count;
    new_pgs->pop = new_pop;
    new_pgs->pool_count = new_pool_count;
    new_pgs->pool = new_pool;

    return new_pgs;
}

// Удаляет перцептронного генетического селекционера.
// В случае успеха возвращает > 0.
// В случае ошибки возвращает < 0.
ptrdiff_t c_pgs_delete(c_pgs *const _pgs)
{
    if (_pgs == NULL)
    {
        return -1;
    }

    for (size_t p = 0; p < _pgs->pool_count; ++p)
    {
        free(_pgs->pool[p].weights);
    }
    for (size_t p = 0; p < _pgs->pop_count; ++p)
    {
        free(_pgs->pop[p].weights);
    }
    free(_pgs->pool);
    free(_pgs->pop);
    free(_pgs->topology);

    free(_pgs);

    return 1;
}

// Запускает процесс обучения заданного перцептрона.
// Селекционер должен быть совместим с перцептроном.
// Уроки должны храниться в виде: ins outs ins outs...
// Уроки должны хранить достаточное количество сигналов.
// В случае успеха возвращает > 0, перцептрон меняет состояние весов.
// В случае ошибки возвращает < 0, перцептрон не меняет состояние весов.
ptrdiff_t c_pgs_run(c_pgs *const _pgs,
                    c_perceptron *const _perceptron,
                    const float *const _lessons,
                    const size_t _lessons_count,
                    const size_t _iterations_count,
                    const float _noise_force,
                    const float _mut_force,
                    uint64_t *const _seed)
{
    if (_pgs == NULL)
    {
        return -1;
    }
    if (_perceptron == NULL)
    {
        return -2;
    }

    // Количество слоев в _perceptron и в _pgs должно совпадать.
    if (_pgs->layers_count != _perceptron->layers_count)
    {
        return -3;
    }

    // Топологии _perceptron и _pgs должны совпадать.
    for (size_t l = 0; l < _pgs->layers_count; ++l)
    {
        if (_pgs->topology[l] != _perceptron->topology[l])
        {
            return -4;
        }
    }

    // Уроки должны быть заданы.
    if (_lessons == NULL)
    {
        return -5;
    }

    // Уроков должно быть больше нуля.
    if (_lessons_count == 0)
    {
        return -6;
    }

    // Итераций должно быть минимум 10.
    if (_iterations_count < 10)
    {
        return -7;
    }

    // Зерно должно быть задано.
    if (_seed == NULL)
    {
        return -8;
    }

    // Определяем количество входных сигналов перцептрона.
    const size_t ins_count = _perceptron->topology[0];
    // Определяем количество выходных сигналов перцептрона.
    const size_t outs_count = _perceptron->topology[_pgs->layers_count - 1];
    // Определяем сумму количества входных и выходных сигналов целевого перцептрона.
    const size_t ins_outs_count = ins_count + outs_count;
    // Контроль целочисленного переполнения при сложении не нужен, так как он
    // осуществляется на этапе конструирования перцептрона, на основе которого конструируется pgs.

    // Определим, сколько памяти занимают входные и выходные сигналы в сумме.
    const size_t ins_outs_size = sizeof(float) * ins_outs_count;
    // Контроль целочисленного переполнения не нужен, так как он
    // осуществляется на этапе конструирования перцептрона.

    // Определим, сколько памяти занимают все уроки, переданные нам.
    const size_t lessons_size = ins_outs_size * _lessons_count;
    // Контроль целочисленного переполнения при умножении.
    if ( (lessons_size == 0) ||
         (lessons_size / ins_outs_size != _lessons_count) )
    {
        return -9;
    }

    // Нужно ли контролировать возможное переполнение указателя при навигации по урокам?
    // Стандарт крайне невнятно описывает это.
    // ...

    // Заполняем начальную популяцию.

    // Одна особь популяции обменивается геномом с заданным перцептроном.
    float_ptr_swap(&_perceptron->weights, &_pgs->pop[0].weights);
    // Геномы остальных особей заполняются шумом.
    for (size_t p = 1; p < _pgs->pop_count; ++p)
    {
        weights_noise(_pgs->pop[p].weights, _perceptron->weights_count, _noise_force, _seed);
    }

    // Выполняем итерации генетического алгоритма:
    // - Скрещивание предков и добавление мутаций;
    // - Тестирование каждого потомка на заданных уроках, подставляя геном потомка в заданный перцептрон;
    // - Сортировка потомков по возрастанию их суммарной ошибки;
    // - Перенос геномов лучших потомков в популяцию;
    // - Повтор.
    for (size_t i = 0; i < _iterations_count; ++i)
    {
        // Скрещиваем геномы особей популяции.
        size_t p3 = 0;
        for (size_t p1 = 0; p1 < _pgs->pop_count; ++p1)
        {
            for (size_t p2 = 0; p2 < _pgs->pop_count; ++p2)
            {
                if (p1 != p2)
                {
                    weights_cross_and_mut(_pgs->pop[p1].weights,
                                          _pgs->pop[p2].weights,
                                          _pgs->pool[p3++].weights,
                                          _perceptron->weights_count,
                                          _mut_force,
                                          _seed);
                }
            }
        }

        // Обходим каждого потомка.
        for (size_t p = 0; p < _pgs->pool_count; ++p)
        {
            // Обмениваем местами веса потомка и веса перцептрона.
            float_ptr_swap(&_perceptron->weights, &_pgs->pool[p].weights);
            // Сбрасываем ошибку потомка.
            _pgs->pool[p].sigma = 0.f;

            // Обходим все уроки.
            for (size_t l = 0; l < _lessons_count; ++l)
            {
                const float *const l_ins = &_lessons[l * ins_outs_count];
                const float *const l_outs = &_lessons[l * ins_outs_count + ins_count];

                // Помещаем входные сигналы урока на вход перцептрона.
                memcpy(_perceptron->ins, l_ins, sizeof(float) * ins_count);

                // Пропускаем сигнал через перцептрон.
                c_perceptron_execute(_perceptron);

                // Вычисляем суммарную ошибку по всем выходным сигналам.
                for (size_t o = 0; o < outs_count; ++o)
                {
                    _pgs->pool[p].sigma += fabs(l_outs[o] - _perceptron->outs[o]);
                }
            }

            // Возвращаем геномы на места.
            float_ptr_swap(&_perceptron->weights, &_pgs->pool[p].weights);
        }

        // Сортируем массив сущностей по возрастанию ошибки.
        qsort(_pgs->pool, _pgs->pool_count, sizeof(c_weights_and_sigma), comp);

        // Отбираем из пула столько лучших, чтобы полностью заполнить популяцию.
        for (size_t p = 0; p < _pgs->pop_count ; ++p)
        {
            float_ptr_swap(&_pgs->pop[p].weights, &_pgs->pool[p].weights);
        }

        // Показываем суммарную ошибку самой умной сети.
        //printf("sigma: %f\n", _pgs->pool[0].sigma);
    }

    // Свопаем веса (геном) перцептрона с весами (геномом) лучшей особи популяции.
    float_ptr_swap(&_perceptron->weights, &_pgs->pop[0].weights);

    return 1;
}
