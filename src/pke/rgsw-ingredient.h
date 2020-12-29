#ifndef CRYPTO_CUDA_RGSW_INGREDIENTS_H
#define CRYPTO_CUDA_RGSW_INGREDIENTS_H

#include "../lattice/CuCRTPoly.h"
#include "../lattice/CuCRTColumnPoly.h"
#include "../lattice/CuCRTParams.h"
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

class CuCRT_RGSW_CryptoParameters {

public:
	CuCRT_RGSW_CryptoParameters();
	CuCRT_RGSW_CryptoParameters(const shared_ptr<CuCRTParams> params, uint32_t p, uint32_t r=1);
	shared_ptr<CuCRTParams> GetElementParams();
	uint32_t GetPlaintextModulus();
	uint32_t GetRelinWindow();
	uint32_t *m_powersOf2 = nullptr;
	uint32_t GetDigits();
	uint32_t GetRows();

private:
	shared_ptr<CuCRTParams> m_params;
	uint32_t m_ptxtModulus;
	uint32_t m_relinWindow;
};

class CuCRT_RGSW_Key {
public:
	CuCRT_RGSW_Key();
	CuCRT_RGSW_Key(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);
	shared_ptr<CuCRT_RGSW_CryptoParameters> GetCryptoParameters() const;

private:
	shared_ptr<CuCRT_RGSW_CryptoParameters> m_cryptoParams;
};

class CuCRT_RGSW_SecretKey : public CuCRT_RGSW_Key {
public:
	CuCRT_RGSW_SecretKey(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);
	const CuCRTPoly& GetSecretKeyElement();
	void SetSecretKeyElement(const CuCRTPoly &poly);
	void SetSecretKeyElement(CuCRTPoly &&poly);

private:
	CuCRTPoly m_sk;

};

class CuCRT_RGSW_PublicKey : public CuCRT_RGSW_Key {
public:
	CuCRT_RGSW_PublicKey(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);

	void SetPublicKeyElementA(const CuCRTPoly &poly);
	void SetPublicKeyElementA(CuCRTPoly &&poly);
	const CuCRTPoly& GetPublicKeyElementA();

	void SetPublicKeyElementB(const CuCRTPoly &poly);
	void SetPublicKeyElementB(CuCRTPoly &&poly);
	const CuCRTPoly& GetPublicKeyElementB();
private:
	CuCRTPoly m_A;
	CuCRTPoly m_B;
};

class CuCRT_RGSW_KeyPair {

public:
	shared_ptr<CuCRT_RGSW_SecretKey> secretKey;
	shared_ptr<CuCRT_RGSW_PublicKey> publicKey;

	CuCRT_RGSW_KeyPair(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);

};

class CuCRT_RGSW_Ciphertext : public CuCRT_RGSW_Key {
public:
	CuCRT_RGSW_Ciphertext(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);
	CuCRT_RGSW_Ciphertext(const CuCRT_RGSW_Ciphertext &cipher);
	CuCRT_RGSW_Ciphertext(CuCRT_RGSW_Ciphertext &&cipher);

	void SetElementA(const CuCRTColumnPoly &colmnA);
	void SetElementA(const CuCRTColumnPoly &&colmnA);
	const CuCRTColumnPoly& GetElementA() const;

	void SetElementB(const CuCRTColumnPoly &colmnB);
	void SetElementB(const CuCRTColumnPoly &&colmnB);
	const CuCRTColumnPoly& GetElementB() const;
public:
	//we assume that m_A and m_B are always in Eval domain and
	//number of rows and crt dimensions can be extracted from cryptoParams
	CuCRTColumnPoly m_A;
	CuCRTColumnPoly m_B;
};

class CuCRT_RGSW_KeySwitchMatrix : public CuCRT_RGSW_Key {

public:
	CuCRT_RGSW_KeySwitchMatrix(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);

public:
	shared_ptr<CuCRT_RGSW_Ciphertext> m_eval0 = nullptr;
	shared_ptr<CuCRT_RGSW_Ciphertext> m_eval1 = nullptr;
};


#endif /* RGSW_INGREDIENT_H_ */
