#ifndef LBCRYPTO_CUPOLY_PARAMS_DEFS_H
#define LBCRYPTO_CUPOLY_PARAMS_DEFS_H

#include "palisade.h"
#include "../math/cubinint.h"
using namespace std;
using namespace lbcrypto;

class CuPolyParams
{

public:
    CuPolyParams(){}
    CuPolyParams(usint m, const BigInteger &modulus, const BigInteger &rootOfUnity);
    usint GetRingDimension() const;
    usint GetCyclotomicNumber() const;
    const BigInteger& GetModulus() const;
    const BigInteger& GetRootOfUnity() const;

    CuBigInteger *m_devModulus = nullptr;
	CuBigInteger *m_devMask = nullptr;
	CuBigInteger *m_devMu = nullptr;
	CuBigInteger *m_devRootOfUnity = nullptr;

private:
    usint m_ringDimension;
    usint m_cycloNumber;
    BigInteger m_modulus;
    BigInteger m_rootOfUnity;
};

#endif
