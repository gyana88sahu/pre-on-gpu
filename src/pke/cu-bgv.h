#ifndef CRYPTO_CUDA_BGV_PKE_H
#define CRYPTO_CUDA_BGV_PKE_H

#include "bgv-ingredient.h"
#include <cuda.h>

using namespace std;

class CuBGV {
public:

	static CuCRT_BGV_KeyPair KeyGen(const shared_ptr<BGV_CryptoParameters> cryptoParams);
	static CuCRT_BGV_Ciphertext Encrypt(const shared_ptr<CuCRT_BGV_PublicKey> pk, const CuCRTPoly &m);
	static std::vector<uint32_t> Decrypt(const CuCRT_BGV_Ciphertext &cipher, const shared_ptr<CuCRT_BGV_SecretKey> sk);

	static CuCRT_BGV_Ciphertext EvalAdd(const CuCRT_BGV_Ciphertext &cipher1,const CuCRT_BGV_Ciphertext &cipher2);
	static CuCRT_BGV_Ciphertext EvalMult(const CuCRT_BGV_Ciphertext &cipher1,const CuCRT_BGV_Ciphertext &cipher2);
	static std::vector<uint32_t*> ReKeyGen(const shared_ptr<CuCRT_BGV_PublicKey> newPk, const shared_ptr<CuCRT_BGV_SecretKey> origPrivateKey);
	static CuCRT_BGV_Ciphertext ReEncrypt(const std::vector<uint32_t*> &rk, const CuCRT_BGV_Ciphertext &cipher);

};

#endif /* BGV_H_ */
