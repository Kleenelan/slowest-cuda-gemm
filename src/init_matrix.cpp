#include "init_matrix.hpp"
void srand_rand_float(int seed, float* A, unsigned long int len)
{
    parallel_mt19937<312>::GetInstance()->srand(seed);
    parallel_mt19937<312>::GetInstance()->rand_float(A, len);
}

void srand_rand_double(int seed, double* A, unsigned long int len)
{
    parallel_mt19937<312>::GetInstance()->srand(seed);
    parallel_mt19937<312>::GetInstance()->rand_double(A, len);
}
