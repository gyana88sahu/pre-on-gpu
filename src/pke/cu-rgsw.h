#ifndef CRYPTO_CUDA_RGSW_PKE_H
#define CRYPTO_CUDA_RGSW_PKE_H

#include "rgsw-ingredient.h"
#include "cu-bgv.h"
#include <cuda.h>

using namespace std;

class CuRGSW {
public:

	/*< PKE OPERATIONS >
	 ----------------*/
	static CuCRT_RGSW_KeyPair KeyGen(const shared_ptr<CuCRT_RGSW_CryptoParameters> cryptoParams);
	static CuCRT_RGSW_Ciphertext Encrypt(const shared_ptr<CuCRT_RGSW_PublicKey> pk, const CuCRTPoly &m);
	static std::vector<uint32_t> Decrypt(const CuCRT_RGSW_Ciphertext &cipher, const shared_ptr<CuCRT_RGSW_SecretKey> sk);

	/*< SHE OPERATIONS >
	 ----------------*/
	static CuCRT_RGSW_Ciphertext EvalAdd(const CuCRT_RGSW_Ciphertext &cipher1,const CuCRT_RGSW_Ciphertext &cipher2);
	static CuCRT_RGSW_Ciphertext EvalMult(const CuCRT_RGSW_Ciphertext &cipherLow,const CuCRT_RGSW_Ciphertext &cipherHigh);
	static CuCRT_BGV_Ciphertext EvalMult(const CuCRT_RGSW_Ciphertext &cipherLow, const CuCRT_BGV_Ciphertext &cipherHigh);

	/*< PRE OPERATIONS >
	 ----------------*/
	static CuCRT_RGSW_KeySwitchMatrix ReKeyGen(const shared_ptr<CuCRT_RGSW_PublicKey> newPk, const shared_ptr<CuCRT_RGSW_SecretKey> origPrivateKey);
	static CuCRT_RGSW_Ciphertext ReEncrypt(const CuCRT_RGSW_KeySwitchMatrix &rk, const CuCRT_RGSW_Ciphertext &cipher);

};

#endif /* BGV_H_ */
