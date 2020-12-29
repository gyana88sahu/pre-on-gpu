#ifndef CRYPTO_CUDA_BGV_INGREDIENTS_H
#define CRYPTO_CUDA_BGV_INGREDIENTS_H

#include "../lattice/CuCRTPoly.h"
#include "../lattice/CuCRTParams.h"
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

class BGV_CryptoParameters {

public:
	BGV_CryptoParameters();
	BGV_CryptoParameters(const shared_ptr<CuCRTParams> params, uint32_t p, uint32_t r=1);
	shared_ptr<CuCRTParams> GetElementParams();
	uint32_t GetPlaintextModulus();
	uint32_t GetRelinWindow();
	uint32_t *m_powersOf2 = nullptr;

private:
	shared_ptr<CuCRTParams> m_params;
	uint32_t m_ptxtModulus;
	uint32_t m_relinWindow;
};

class CuCRT_BGV_Key {
public:
	CuCRT_BGV_Key();
	CuCRT_BGV_Key(const shared_ptr<BGV_CryptoParameters> params);
	shared_ptr<BGV_CryptoParameters> GetCryptoParameters() const;

private:
	shared_ptr<BGV_CryptoParameters> m_cryptoParams;
};

class CuCRT_BGV_SecretKey : public CuCRT_BGV_Key {
public:
	CuCRT_BGV_SecretKey(const shared_ptr<BGV_CryptoParameters> params);
	const CuCRTPoly& GetSecretKeyElement();
	void SetSecretKeyElement(const CuCRTPoly &poly);
	void SetSecretKeyElement(CuCRTPoly &&poly);

private:
	CuCRTPoly m_sk;

};

class CuCRT_BGV_PublicKey : public CuCRT_BGV_Key {
public:
	CuCRT_BGV_PublicKey(const shared_ptr<BGV_CryptoParameters> params);

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

class CuCRT_BGV_KeyPair {

public:
	shared_ptr<CuCRT_BGV_SecretKey> secretKey;
	shared_ptr<CuCRT_BGV_PublicKey> publicKey;

	CuCRT_BGV_KeyPair(const shared_ptr<BGV_CryptoParameters> params);

};

class CuCRT_BGV_Ciphertext : public CuCRT_BGV_Key {
public:
	CuCRT_BGV_Ciphertext(const shared_ptr<BGV_CryptoParameters> params);
	CuCRT_BGV_Ciphertext(const CuCRT_BGV_Ciphertext &cipher);
	CuCRT_BGV_Ciphertext(CuCRT_BGV_Ciphertext &&cipher);

	void SetElementA(const CuCRTPoly &poly);
	void SetElementA(CuCRTPoly &&poly);
	const CuCRTPoly& GetElementA() const;

	void SetElementB(const CuCRTPoly &poly);
	void SetElementB(CuCRTPoly &&poly);
	const CuCRTPoly& GetElementB() const;
private:
	CuCRTPoly m_A;
	CuCRTPoly m_B;
};

#endif /* BGV_INGREDIENT_H_ */
