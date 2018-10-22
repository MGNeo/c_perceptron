#include <stdio.h>
#include <stdlib.h>

#include "c_perceptron.h"

int main(int argc, char **argv)
{
    size_t error;
    c_perceptron *perceptron;

    // Для работы внутреннего ГПСЧ перцептрону и селекционеру необходимо предоставить зерно.
    // Если операции, требующие предоставить зерно, будут выполняться в многопоточной среде,
    // каждый поток должен обладать своим используемым зерном.
    uint64_t seed = 1;

    // Создаем перцептрон.
    // Первый слой перцептрона - это слой входов.
    // Остальные слои перцептрона являются активными (состоят из нейронов).
    // Перцептрон содержит столько выходов, сколько нейронов имеется в его последнем слое.
    {
        const size_t topology[4] = {1, 5, 8, 1};
        perceptron = c_perceptron_create(4, topology, &error);

        // Если не удалось создать перцептрон, показываем код ошибки.
        if (perceptron == NULL)
        {
            printf("c_perceptron_create() error: %Iu\n", error);
            getchar();
            return -1;
        }
    }

    // Заполняем веса перцептрона начальным шумом [-1.f; +1.f].
    {
        const ptrdiff_t r_code = c_perceptron_noise(perceptron, 1.f, &seed);
        // Если не удалось заполнить перцептрон шумом, показываем код ошибки.
        if (r_code < 0)
        {
            c_perceptron_delete(perceptron);
            printf("c_perceptron_noise() error: %Id\n", r_code);
            getchar();
            return -2;
        }
    }

    // Создаем перцептронного генетического селекционера.
    // pgs создается на основе перцептрона потому, что pgs должен знать количество весов и топологию сетей,
    // с которыми он будет работать.
    // pgs может использовать любой перцептрон, топология которого идентична топологии перцептрона, на основе
    // которого pgs был создан.
    // При размножении pgs будет использовать популяцию из 20 особей.
    c_pgs *pgs;
    {
        pgs = c_pgs_create(perceptron, 20, &error);
        // Если не удалось создать pgs, показываем код ошибки.
        if (pgs == NULL)
        {
            c_perceptron_delete(perceptron);
            printf("c_pgs_create() error: %Iu\n", error);
            getchar();
            return -3;
        }
    }

    // Уроки будут обучать перцептрон находить квадрат чисел из диапазона [+0.1f; +0.9f].
    float lessons[18] = {0.1, 0.01,
                         0.2, 0.04,
                         0.3, 0.09,
                         0.4, 0.16,
                         0.5, 0.25,
                         0.6, 0.36,
                         0.7, 0.49,
                         0.8, 0.64,
                         0.9, 0.81};

    // Запустим цикл обучения перцептрона селекционером на девяти уроках.
    // Цикл будет состоять из 1000 итераций. Количество итераций не должно быть меньше 10.
    // Сила шума определяет шумовое заполнение весов начальной популяции [-_noise_force, +_noise_force].
    // Сила мутации влияет на максимальную величину мутации (изменения веса) при скрещивании [-_mut_force; +_mut_force].
    {
        printf("Learning...\n");

        const ptrdiff_t r_code = c_pgs_run(pgs, perceptron, lessons, 9, 1000, 1.f, 1.f, &seed);
        // Если не удалось запустить процесс обучения, показываем код ошибки.
        if (r_code < 0)
        {
            c_pgs_delete(pgs);
            c_perceptron_delete(perceptron);
            printf("c_pgs_run() error: %Id\n", r_code);
            getchar();
            return -4;
        }
    }

    // Сохраняем перцептрон в файл.
    {
        const ptrdiff_t r_code = c_perceptron_save(perceptron, "perceptron");
        // Если сохранить перцептрон в файл не удалось, покажем код ошибки.
        if (r_code < 0)
        {
            c_pgs_delete(pgs);
            c_perceptron_delete(perceptron);
            printf("c_perceptron_save() error: %Id\n", r_code);
            getchar();
            return -5;
        }
    }

    // Загружаем перцептрон из файла.
    c_perceptron *loaded_perceptron;
    {
        loaded_perceptron = c_perceptron_load("perceptron", &error);
        // Если загрузить перцептрон из файла не удалось, покажем код ошибки.
        if (loaded_perceptron == NULL)
        {
            c_pgs_delete(pgs);
            c_perceptron_delete(perceptron);
            printf("c_perceptron_load() error: %Iu\n", error);
            getchar();
            return -6;
        }
    }

    // Тестируем перцептрон, загруженный из файла.

    // Получаем прямой доступ ко входам перцептрона.
    float *const ins = c_perceptron_get_ins(loaded_perceptron);
    if (ins == NULL)
    {
        c_perceptron_delete(loaded_perceptron);
        c_pgs_delete(pgs);
        c_perceptron_delete(perceptron);
        printf("c_perceptron_get_ins() error.\n");
        getchar();
        return -7;
    }

    // Получаем прямой доступ к выходам перцептрона.
    const float *const outs = c_perceptron_get_outs(loaded_perceptron);
    if (outs == NULL)
    {
        c_perceptron_delete(loaded_perceptron);
        c_pgs_delete(pgs);
        c_perceptron_delete(perceptron);
        printf("c_perceptron_get_outs() error.\n");
        getchar();
        return -8;
    }

    for (size_t i = 0; i < 10; ++i)
    {
        // Генерируем какой-то входной сигнал.
        const float in_signal = (rand() % 100) / 100.f;

        // Помещаем один входной сигнал на единственный вход перцептрона.
        ins[0] = in_signal;

        // Пропускаем сигнал через перцептрон.
        c_perceptron_execute(loaded_perceptron);

        // Покажем входной и выходной сигналы (прямо с перцептрона).
        printf("in: %f out: %f\n", ins[0], outs[0]);
    }

    // Удаляем перцептрон, загруженный из файла.
    c_perceptron_delete(loaded_perceptron);

    // Удаляем перцептронного генетического селекционера.
    c_pgs_delete(pgs);

    // Удаляем перцептрон.
    c_perceptron_delete(perceptron);

    return 0;
}
