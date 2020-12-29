#include "CuPolyParams.h"

CuPolyParams::CuPolyParams(usint m, const BigInteger &modulus, const BigInteger &rootOfUnity){
    this->m_ringDimension = m/2;
    this->m_cycloNumber = m;
    this->m_modulus = modulus;
    this->m_rootOfUnity = rootOfUnity;
    BigInteger mask(1);
	mask <<= m_modulus.GetMSB();
	mask -= BigInteger(1);
	auto mu = ComputeMu(m_modulus);
    CuBigInteger::InitializeDeviceVariables(&m_devModulus, m_modulus.ToString());
    CuBigInteger::InitializeDeviceVariables(&m_devRootOfUnity, m_rootOfUnity.ToString());
	CuBigInteger::InitializeDeviceVariables(&m_devMask, mask.ToString());
	CuBigInteger::InitializeDeviceVariables(&m_devMu, mu.ToString());
}
usint CuPolyParams::GetRingDimension() const {
    return this->m_ringDimension;
}

usint CuPolyParams::GetCyclotomicNumber() const {
    return this->m_cycloNumber;
}

const BigInteger& CuPolyParams::GetModulus() const {
    return this->m_modulus;
}

const BigInteger& CuPolyParams::GetRootOfUnity() const{
    return this->m_rootOfUnity;
}
