#ifndef LBCRYPTO_CU_CRT_POLY_PARAMS_DEFS_H
#define LBCRYPTO_CU_CRT_POLY_PARAMS_DEFS_H

#include "palisade.h"
#include "../math/cubinint.h"
#include <cmath>
#include <curand.h>
#include <cuda.h>

using namespace std;
using namespace lbcrypto;

//extern __constant__ uint32_t modulii2[20];

class CuCRTParams{

public:
	CuCRTParams() {};
	CuCRTParams(usint m, const std::vector<uint32_t> &modulii, const std::vector<uint32_t> &rootOfUnities);
	usint GetRingDimension() const;
	usint GetCyclotomicNumber() const;
	const BigInteger& GetModulus() const; //does the crt inverse

	uint32_t *m_devModulii = nullptr;
	uint32_t *m_devRootOfUnities = nullptr;
	uint32_t *m_devInverseRootOfUnities = nullptr;
	uint32_t *m_devModuliiMask = nullptr;
	uint32_t *m_devMus = nullptr;


	CuBigInteger* m_devBigModulus = nullptr;
	CuBigInteger* m_devBigMu = nullptr;
	uint32_t* m_bxis = nullptr;

	std::vector<uint32_t> m_hostModulii;
	std::vector<uint32_t> m_hostRootOfUnities;
	curandGenerator_t m_dug;
	curandGenerator_t m_dgg;

	//store the big modulus
	BigInteger bigModulus;


private:
	usint m_ringDimension;
	usint m_cycloNumber;
	BigInteger m_modulus;

};

#endif
