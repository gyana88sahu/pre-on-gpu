#ifndef LBCRYPTO_CUBIGVECT_DEFS_H
#define LBCRYPTO_CUBIGVECT_DEFS_H

#include "cubinint.h"

using namespace std;

class CuBigVector{

  public:

	__host__ __device__ CuBigVector();

  __host__ __device__ CuBigVector(uint32_t length, uint32_t *modulus);

  __host__ __device__ CuBigVector(const CuBigVector &bigVector);    

  __device__ CuBigVector ModAdd(const CuBigVector &a, const CuBigVector &b);

	uint32_t *m_data;

	uint32_t m_length = 0;

  uint32_t *m_modulus;

};

#endif