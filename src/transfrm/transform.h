#ifndef LBCRYPTO_CUTRFM_DEFS_H
#define LBCRYPTO_CUTRFM_DEFS_H

#include "palisade.h"
#include "../math/cubinint.h"
#include "../math/nativeint.h"
#include <curand.h>
#include <cuda.h>

using namespace lbcrypto;

//extern __constant__ uint32_t modulii2[20];

class CuChineseRemainderTransformFTT{

public:
	CuChineseRemainderTransformFTT(){}
	static void ForwardTransform(uint32_t *inData, const BigInteger &modulus, const BigInteger &root, usint cycloOrder, uint32_t *outData );
	static void InverseTransform(uint32_t *inData, const BigInteger &modulus, const BigInteger &root, usint cycloOrder, uint32_t *outData );
	static void ComputeBitReverseTable(usint n);
	static uint32_t *m_rTableDev;
	static uint32_t *m_rInverseTableDev;
	static CuBigInteger *devModulus;
	static CuBigInteger *devRoot;
	static CuBigInteger *devRootInverse;
	static CuBigInteger *devMu;
	static uint32_t *bitReverseTable;
};


class CuCRTChineseRemainderTransformFTT{

public:
	CuCRTChineseRemainderTransformFTT(){}
	static void ForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, cudaStream_t *stream);
	static void InverseTransform(uint32_t *inData, uint32_t *modulii, uint32_t *rootIs, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, cudaStream_t *stream);
	static void RelinForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint polySize,
			cudaStream_t *stream);

	static void ColumnForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint polySize,
			cudaStream_t &stream);

	static void ColumnInverseTransform(uint32_t *inData, uint32_t *modulii, uint32_t *rootIs, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint rows,
				cudaStream_t &stream);
	static void ComputeBitReverseTable(usint n);
	static uint32_t *m_rTableDev;
	static uint32_t *m_rInverseTableDev;
	static uint32_t *bitReverseTable;
	static uint32_t *m_nInverseTable;
};

#endif
