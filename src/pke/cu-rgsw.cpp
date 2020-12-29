#include "cu-rgsw.h"

static __global__ void EncryptKernel( uint32_t *colmnData, uint32_t *polyData, uint32_t rowOffset, uint32_t* powersOf2, uint32_t *modulii, uint32_t *mus);

static __global__ void ReKeyGenKernelUpper(uint32_t* cipherData, uint32_t* powersOf2, uint32_t *modulii);

static __global__ void ReKeyGenKernelLower(uint32_t* cipherData, uint32_t* skData, uint32_t* powersOf2, uint32_t *modulii, uint32_t *mus);

static __global__ void EvalAddKernel(uint32_t* cipherOut, uint32_t* cipherIn1, uint32_t* cipherIn2, uint32_t *modulii);

CuCRT_RGSW_KeyPair CuRGSW::KeyGen(const shared_ptr<CuCRT_RGSW_CryptoParameters> cryptoParams){

	CuCRT_RGSW_KeyPair kp(cryptoParams);

	auto ep = cryptoParams->GetElementParams();
	auto p = cryptoParams->GetPlaintextModulus();

	//creating 3 streams for s,a, and e
	cudaStream_t sStream, aStream, eStream;
	cudaStreamCreate(&sStream);
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&eStream);

	CuCRTPoly s(ep, CuCRTPoly::Format::COEFFICIENT, CuCRTPoly::NoiseType::TERNARY, &sStream);
	s.SwitchFormat(&sStream); // s is now in EVAL domain

	//public key generations

	//generate uniform noise a
	CuCRTPoly a(ep, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::UNIFORM, &aStream);
	//generate gaussian error e
	CuCRTPoly e(ep, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::GAUSSIAN, &eStream);

	cudaStreamSynchronize(sStream);
	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(eStream);

	cudaStreamDestroy(sStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(eStream);

	auto b(a*s + p*e);

	kp.secretKey->SetSecretKeyElement(std::move(s));
	kp.publicKey->SetPublicKeyElementA(std::move(a));
	kp.publicKey->SetPublicKeyElementB(std::move(b));

	return kp;
}

CuCRT_RGSW_Ciphertext CuRGSW::Encrypt(const shared_ptr<CuCRT_RGSW_PublicKey> pk, const CuCRTPoly &m){

	auto params = pk->GetCryptoParameters();
	CuCRT_RGSW_Ciphertext cipher(params);
	auto ep = params->GetElementParams();

	auto p = params->GetPlaintextModulus();
	auto rows = params->GetRows();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();

	cudaStream_t mssgStream;
	cudaStreamCreate(&mssgStream);
	CuCRTPoly mCopy(m);
	mCopy.SwitchFormat(&mssgStream);
	cudaStreamDestroy(mssgStream);

	//creating 3 streams for r,e0, and e1
	cudaStream_t rStream, e0Stream, e1Stream;
	cudaStreamCreate(&rStream);
	cudaStreamCreate(&e0Stream);
	cudaStreamCreate(&e1Stream);

	CuCRTColumnPoly r(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::TERNARY, rows, rStream);
	CuCRTColumnPoly e0(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e0Stream);
	CuCRTColumnPoly e1(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e1Stream);

	//because noises are internally synced we do not explicitly sync them here
	cudaStreamDestroy(rStream);
	cudaStreamDestroy(e0Stream);
	cudaStreamDestroy(e1Stream);

	const auto& a = pk->GetPublicKeyElementA();
	const auto& b = pk->GetPublicKeyElementB();

	cipher.m_A = std::move(r*a);
	cipher.m_A += p*e1;

	cipher.m_B = std::move(r*b);
	cipher.m_B += p*e0;

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(rows>>1, crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(rows>>1, crtLength, dim >> 10);
		block = dim3(1024);
	}
	EncryptKernel<<<grid,block,0,bStream>>>(cipher.m_B.m_data, mCopy.m_data, 0, params->m_powersOf2, ep->m_devModulii, ep->m_devMus);
	EncryptKernel<<<grid,block,0,aStream>>>(cipher.m_A.m_data, mCopy.m_data, rows/2, params->m_powersOf2, ep->m_devModulii, ep->m_devMus);


	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);
	return std::move(cipher);
}

std::vector<uint32_t> CuRGSW::Decrypt(const CuCRT_RGSW_Ciphertext &cipher, const shared_ptr<CuCRT_RGSW_SecretKey> sk){
	auto a(std::move(cipher.m_A[0]));
	auto b(std::move(cipher.m_B[0]));

	auto p = cipher.GetCryptoParameters()->GetPlaintextModulus();

	const auto& s = sk->GetSecretKeyElement();
	cudaStream_t decStream;
	cudaStreamCreate(&decStream);

	auto as(a*s);
	auto noise(b-as);
	noise.SwitchFormat(&decStream);

	auto ptxt = noise.SignedMod(p);

	cudaStreamDestroy(decStream);
	return ptxt;
}

CuCRT_RGSW_Ciphertext CuRGSW::EvalAdd(const CuCRT_RGSW_Ciphertext &cipher1,const CuCRT_RGSW_Ciphertext &cipher2){
	auto params = cipher1.GetCryptoParameters();
	CuCRT_RGSW_Ciphertext cipherResult(params);
	auto ep = params->GetElementParams();

	auto p = params->GetPlaintextModulus();
	auto rows = params->GetRows();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(rows, crtLength);
		block = dim3(dim);
	}
	else {
		grid = dim3(rows, crtLength, dim >> 10);
		block = dim3(1024);
	}

	EvalAddKernel<<<grid,block,0, aStream>>>(cipherResult.m_A.m_data, cipher1.m_A.m_data, cipher2.m_A.m_data, ep->m_devModulii);
	EvalAddKernel<<<grid,block,0, bStream>>>(cipherResult.m_B.m_data, cipher1.m_B.m_data, cipher2.m_B.m_data, ep->m_devModulii);

	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);

	return std::move(cipherResult);
}

CuCRT_RGSW_Ciphertext CuRGSW::EvalMult(const CuCRT_RGSW_Ciphertext &cipherLow,const CuCRT_RGSW_Ciphertext &cipherHigh){
	//high will be bitdecomposed and low is in power of 2 form
	auto params = cipherLow.GetCryptoParameters();
	CuCRT_RGSW_Ciphertext cipherResult(params);
	auto ep = params->GetElementParams();

	auto p = params->GetPlaintextModulus();
	auto rows = params->GetRows();
	auto relinWindow = params->GetRelinWindow();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();
	auto polySize = dim*crtLength;

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	for (uint32_t i = 0; i < rows; i++) {
		auto bPoly(std::move(cipherHigh.m_B[i]));
		auto aPoly(std::move(cipherHigh.m_A[i]));
		auto colmnPoly = std::move(CuCRTColumnPoly::GetBitDecomposedColumnPoly(bPoly, aPoly, rows, relinWindow));
		auto polyBresult = (colmnPoly*cipherLow.m_B).Reduce(bStream);
		auto polyAresult = (colmnPoly*cipherLow.m_A).Reduce(aStream);
		uint32_t *offsetRowPtrA = cipherResult.m_A.m_data + dim*crtLength*i;
		uint32_t *offsetRowPtrB = cipherResult.m_B.m_data + dim*crtLength*i;
		cudaMemcpyAsync(offsetRowPtrA, polyAresult.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, aStream);
		cudaMemcpyAsync(offsetRowPtrB, polyBresult.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, bStream);
	}

	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);

	return std::move(cipherResult);
}

CuCRT_BGV_Ciphertext CuRGSW::EvalMult(const CuCRT_RGSW_Ciphertext &cipherLow, const CuCRT_BGV_Ciphertext &cipherHigh){
	//high will be bitdecomposed and low is in power of 2 form
	auto paramsRGSW = cipherLow.GetCryptoParameters();
	auto paramsBGV = cipherHigh.GetCryptoParameters();
	CuCRT_BGV_Ciphertext cipherResult(paramsBGV);
	auto ep = paramsRGSW->GetElementParams();

	auto p = paramsRGSW->GetPlaintextModulus();
	auto rows = paramsRGSW->GetRows();
	auto relinWindow = paramsRGSW->GetRelinWindow();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();
	auto polySize = dim*crtLength;

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	auto colmnPoly = std::move(CuCRTColumnPoly::GetBitDecomposedColumnPoly(cipherHigh.GetElementB(), cipherHigh.GetElementA(), rows, relinWindow));
	auto polyBresult = (colmnPoly*cipherLow.m_B).Reduce(bStream);
	auto polyAresult = (colmnPoly*cipherLow.m_A).Reduce(aStream);

	cipherResult.SetElementA(std::move(polyAresult));
	cipherResult.SetElementB(std::move(polyBresult));

	return std::move(cipherResult);
}


CuCRT_RGSW_KeySwitchMatrix CuRGSW::ReKeyGen(const shared_ptr<CuCRT_RGSW_PublicKey> newPk, const shared_ptr<CuCRT_RGSW_SecretKey> origPrivateKey){
	auto params = newPk->GetCryptoParameters();
	auto ep = params->GetElementParams();
	auto rows = params->GetRows();
	auto p = params->GetPlaintextModulus();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();

	CuCRT_RGSW_KeySwitchMatrix result(params);

	const auto &bPolySubs = newPk->GetPublicKeyElementB();
	const auto &aPolySubs = newPk->GetPublicKeyElementA();

	const auto &sPolyProd = origPrivateKey->GetSecretKeyElement();

	//creating 3 streams for r,e0, and e1
	cudaStream_t rStream, e0Stream, e1Stream;
	cudaStreamCreate(&rStream);
	cudaStreamCreate(&e0Stream);
	cudaStreamCreate(&e1Stream);

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(rows>>1, crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(rows>>1, crtLength, dim >> 10);
		block = dim3(1024);
	}

	{
		CuCRTColumnPoly r(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::TERNARY, rows, rStream);
		CuCRTColumnPoly e0(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e0Stream);
		CuCRTColumnPoly e1(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e1Stream);
		result.m_eval0->m_A = std::move(r*aPolySubs);
		result.m_eval0->m_A += p*e1;

		result.m_eval0->m_B = std::move(r*bPolySubs);
		result.m_eval0->m_B += p*e0;
	}

	ReKeyGenKernelUpper<<<grid,block,0,aStream>>>(result.m_eval0->m_B.m_data, params->m_powersOf2, ep->m_devModulii);
	ReKeyGenKernelLower<<<grid,block,0,bStream>>>(result.m_eval0->m_B.m_data, sPolyProd.m_data, params->m_powersOf2, ep->m_devModulii, ep->m_devMus);


	{
		CuCRTColumnPoly r(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::TERNARY, rows, rStream);
		CuCRTColumnPoly e0(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e0Stream);
		CuCRTColumnPoly e1(ep, CuCRTColumnPoly::Format::EVALUATION, CuCRTColumnPoly::NoiseType::GAUSSIAN, rows, e1Stream);
		result.m_eval1->m_A = std::move(r*aPolySubs);
		result.m_eval1->m_A += p*e1;

		result.m_eval1->m_B = std::move(r*bPolySubs);
		result.m_eval1->m_B += p*e0;
	}

	ReKeyGenKernelUpper<<<grid,block,0,aStream>>>(result.m_eval1->m_A.m_data, params->m_powersOf2, ep->m_devModulii);
	ReKeyGenKernelLower<<<grid,block,0,bStream>>>(result.m_eval1->m_A.m_data, sPolyProd.m_data, params->m_powersOf2, ep->m_devModulii, ep->m_devMus);

	//because noises are internally synced we do not explicitly sync them here
	cudaStreamDestroy(rStream);
	cudaStreamDestroy(e0Stream);
	cudaStreamDestroy(e1Stream);

	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);
	return std::move(result);
}

CuCRT_RGSW_Ciphertext CuRGSW::ReEncrypt(const CuCRT_RGSW_KeySwitchMatrix &rk, const CuCRT_RGSW_Ciphertext &cipher){
	auto params = cipher.GetCryptoParameters();
	CuCRT_RGSW_Ciphertext cipherNew(params);
	auto ep = params->GetElementParams();

	auto p = params->GetPlaintextModulus();
	auto rows = params->GetRows();
	auto ell = rows >> 1;
	auto relinWindow = params->GetRelinWindow();
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();
	auto polySize = dim*crtLength;
	auto offset = dim*crtLength*ell;

	cudaStream_t aUpperStream, aLowerStream, bUpperStream, bLowerStream;
	cudaStreamCreate(&aUpperStream);
	cudaStreamCreate(&aLowerStream);
	cudaStreamCreate(&bUpperStream);
	cudaStreamCreate(&bLowerStream);

	for (uint32_t i = 0; i < ell; i++) {
		auto bPoly(std::move(cipher.m_B[i]));
		auto aPoly(std::move(cipher.m_A[i]));
		auto colmnPoly = std::move(CuCRTColumnPoly::GetBitDecomposedColumnPoly(bPoly, aPoly,rows, relinWindow));
		auto rowUpperB = (colmnPoly*rk.m_eval0->m_B).Reduce(bUpperStream);
		auto rowUpperA = (colmnPoly*rk.m_eval0->m_A).Reduce(aUpperStream);
		auto rowLowerB = (colmnPoly*rk.m_eval1->m_B).Reduce(bLowerStream);
		auto rowLowerA = (colmnPoly*rk.m_eval1->m_A).Reduce(aLowerStream);
		uint32_t *offsetRowPtrUpperA = cipherNew.m_A.m_data + dim*crtLength*i;
		uint32_t *offsetRowPtrUpperB = cipherNew.m_B.m_data + dim*crtLength*i;
		uint32_t *offsetRowPtrLowerA = cipherNew.m_A.m_data + offset + dim*crtLength*i;
		uint32_t *offsetRowPtrLowerB = cipherNew.m_B.m_data + offset + dim*crtLength*i;
		cudaMemcpyAsync(offsetRowPtrUpperA, rowUpperA.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, aUpperStream);
		cudaMemcpyAsync(offsetRowPtrUpperB, rowUpperB.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, bUpperStream);
		cudaMemcpyAsync(offsetRowPtrLowerA, rowLowerA.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, aLowerStream);
		cudaMemcpyAsync(offsetRowPtrLowerB, rowLowerB.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, bLowerStream);
	}

	cudaStreamSynchronize(aUpperStream);
	cudaStreamSynchronize(aLowerStream);
	cudaStreamSynchronize(bUpperStream);
	cudaStreamSynchronize(bLowerStream);

	cudaStreamDestroy(aUpperStream);
	cudaStreamDestroy(aLowerStream);
	cudaStreamDestroy(bUpperStream);
	cudaStreamDestroy(bLowerStream);

	return std::move(cipherNew);
}

__global__ void EncryptKernel( uint32_t *colmnData, uint32_t *polyData, uint32_t rowOffset, uint32_t* powersOf2, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = (block + rowOffset*gridDim.y*gridDim.z)*blockDim.x + threadIdx.x;

	uint32_t polyIdx = (blockIdx.z + blockIdx.y*gridDim.z)*blockDim.x + threadIdx.x;
	uint32_t powerIdx = blockIdx.x*gridDim.y + blockIdx.y;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	//columnData += polyData*powerof2
	uint32_t mssgPow = ModBarretMult(polyData[polyIdx],powersOf2[powerIdx],modulus,mod2,mu,bits);
	colmnData[idx] = ModAdd(colmnData[idx],mssgPow,modulus);
}

__global__ void ReKeyGenKernelUpper(uint32_t* cipherData, uint32_t* powersOf2, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t powerIdx = blockIdx.x*gridDim.y + blockIdx.y;
	uint32_t modulus = modulii[blockIdx.y];

	cipherData[idx] = ModAdd(cipherData[idx],powersOf2[powerIdx],modulus);
}

__global__ void ReKeyGenKernelLower(uint32_t* cipherData, uint32_t* skData, uint32_t* powersOf2, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t rowOffset = gridDim.x;
	uint32_t idx = (block + rowOffset*gridDim.y*gridDim.z)*blockDim.x + threadIdx.x;

	uint32_t polyIdx = (blockIdx.z + blockIdx.y*gridDim.z)*blockDim.x + threadIdx.x;
	uint32_t powerIdx = blockIdx.x*gridDim.y + blockIdx.y;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	//columnData += polyData*powerof2
	uint32_t skPow = ModBarretMult(skData[polyIdx],powersOf2[powerIdx],modulus,mod2,mu,bits);
	cipherData[idx] = ModSub(cipherData[idx],skPow,modulus);
}

__global__ void EvalAddKernel(uint32_t* cipherOut, uint32_t* cipherIn1, uint32_t* cipherIn2, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];

	cipherOut[idx] = ModAdd(cipherIn1[idx], cipherIn2[idx], modulus);
}

