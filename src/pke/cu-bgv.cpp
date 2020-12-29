
#include "cu-bgv.h"

static __global__ void MultiplyDigitPolynomials(uint32_t *poly, uint32_t* digitPoly, uint32_t* outDigitPoly, uint32_t* modulii, uint32_t* mus);
static __global__ void MultiplyScalarWithDigitPolynomials(uint32_t* digitPoly, uint32_t multVal, uint32_t* modulii, uint32_t* mus);
static __global__ void AddDigitPolynomials(uint32_t* digitPolyIn1, uint32_t* digitPolyIn2, uint32_t* digitPolyOut, uint32_t* modulii);
static __global__ void SubtractDigitPolynomials(uint32_t* digitPolyIn1, uint32_t* digitPolyIn2, uint32_t* digitPolyOut, uint32_t* modulii);
static __global__ void LogAddDigitPolynomials(uint32_t *data, uint32_t cap, uint32_t *modulii);
static __global__ void PolynomialAdditionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii);
static __global__ void MultiplyRelinPolynomials(uint32_t *polyA, uint32_t* polyB, uint32_t* polyOut, uint32_t* modulii, uint32_t* mus);


CuCRT_BGV_KeyPair CuBGV::KeyGen(const shared_ptr<BGV_CryptoParameters> cryptoParams){

	CuCRT_BGV_KeyPair kp(cryptoParams);

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

CuCRT_BGV_Ciphertext CuBGV::Encrypt(const shared_ptr<CuCRT_BGV_PublicKey> pk, const CuCRTPoly &m){

	CuCRT_BGV_Ciphertext cipher(pk->GetCryptoParameters());
	auto ep = pk->GetCryptoParameters()->GetElementParams();

	auto p = pk->GetCryptoParameters()->GetPlaintextModulus();

	cudaStream_t mssgStream;
	cudaStreamCreate(&mssgStream);
	CuCRTPoly mCopy(m);
	mCopy.SwitchFormat(&mssgStream);
	cudaStreamDestroy(mssgStream);

	//creating 3 streams for v,e0, and e1
	cudaStream_t vStream, e0Stream, e1Stream;
	cudaStreamCreate(&vStream);
	cudaStreamCreate(&e0Stream);
	cudaStreamCreate(&e1Stream);

	CuCRTPoly v(ep, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::GAUSSIAN, &vStream);
	CuCRTPoly e0(ep, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::GAUSSIAN, &e0Stream);
	CuCRTPoly e1(ep, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::GAUSSIAN, &e1Stream);


	const auto& a = pk->GetPublicKeyElementA();
	const auto& b = pk->GetPublicKeyElementB();

	//because eval noises are internally synced we do not explicitly sync them here

	cudaStreamDestroy(vStream);
	cudaStreamDestroy(e0Stream);
	cudaStreamDestroy(e1Stream);

	auto cipherElementA(a*v + p*e1);
	auto cipherElementB(b*v + p*e0 + mCopy);

	cipher.SetElementA(std::move(cipherElementA));
	cipher.SetElementB(std::move(cipherElementB));

	return std::move(cipher);
}


std::vector<uint32_t> CuBGV::Decrypt(const CuCRT_BGV_Ciphertext &cipher, const shared_ptr<CuCRT_BGV_SecretKey> sk){
	const auto& a = cipher.GetElementA();
	const auto& b = cipher.GetElementB();

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

CuCRT_BGV_Ciphertext CuBGV::EvalAdd(const CuCRT_BGV_Ciphertext &cipher1,const CuCRT_BGV_Ciphertext &cipher2){
	CuCRT_BGV_Ciphertext result(cipher1.GetCryptoParameters());

	auto a(cipher1.GetElementA()+cipher2.GetElementA());
	auto b(cipher1.GetElementB()+cipher2.GetElementB());

	result.SetElementA(std::move(a));
	result.SetElementB(std::move(b));

	return std::move(result);
}

std::vector<uint32_t*> CuBGV::ReKeyGen(const shared_ptr<CuCRT_BGV_PublicKey> newPK, const shared_ptr<CuCRT_BGV_SecretKey> origPrivateKey){
	uint32_t* c0 = nullptr; //refers to the b-term
	uint32_t* c1 = nullptr; // refers to the a-term
	std::vector<uint32_t*> result;

	auto cryptoParams = newPK->GetCryptoParameters();
	auto elemParams = cryptoParams->GetElementParams();
	auto p = cryptoParams->GetPlaintextModulus();

	usint r = cryptoParams->GetRelinWindow();
	usint nBits = elemParams->bigModulus.GetMSB();
	usint nDigits = 1;
	if (r > 0) {
		nDigits = nBits / r;
		if (nBits % r > 0){
			nDigits++;
		}
	}

	uint32_t relinPolyLength = elemParams->m_hostModulii.size()*nDigits*elemParams->GetRingDimension();
	//allocate memory for c0
	cudaError_t err = cudaMalloc(&c0, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for c0 polynomial\n");
	}
	err = cudaMalloc(&c1, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for c1 polynomial\n");
	}

	const auto& s = origPrivateKey->GetSecretKeyElement();
	const auto& pkA = newPK->GetPublicKeyElementA();
	const auto& pkB = newPK->GetPublicKeyElementB();

	//long polynomial generation starts
	cudaStream_t noiseStreams[3], pow2Stream;
	cudaStreamCreate(&noiseStreams[0]);
	cudaStreamCreate(&noiseStreams[1]);
	cudaStreamCreate(&noiseStreams[2]);
	cudaStreamCreate(&pow2Stream);
	auto powersOfS = s.PowersOf2(r, cryptoParams->m_powersOf2, &pow2Stream);
	auto v = CuCRTPoly::GenerateRelinNoise(elemParams, CuCRTPoly::EVALUATION, CuCRTPoly::TERNARY, nDigits, &noiseStreams[0]);
	auto e0 = CuCRTPoly::GenerateRelinNoise(elemParams, CuCRTPoly::EVALUATION, CuCRTPoly::GAUSSIAN,  nDigits, &noiseStreams[1]);
	auto e1 = CuCRTPoly::GenerateRelinNoise(elemParams, CuCRTPoly::EVALUATION, CuCRTPoly::GAUSSIAN, nDigits, &noiseStreams[2]);

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	usint dim = elemParams->GetRingDimension();
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(nDigits,s.m_crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(nDigits,s.m_crtLength);
		block = dim3(dim);
	}

	cudaStreamDestroy(noiseStreams[0]);
	cudaStreamDestroy(noiseStreams[1]);
	cudaStreamDestroy(noiseStreams[2]);

	//performs c0 = pkB*v
	MultiplyDigitPolynomials<<<grid,block,0,bStream>>>(pkB.m_data, v, c0, elemParams->m_devModulii, elemParams->m_devMus);
	//performs c1 = pkA*v
	MultiplyDigitPolynomials<<<grid,block,0,aStream>>>(pkA.m_data, v, c1, elemParams->m_devModulii, elemParams->m_devMus);


	//performs e0 = p*e0
	MultiplyScalarWithDigitPolynomials<<<grid,block,0,bStream>>>(e0, p, elemParams->m_devModulii, elemParams->m_devMus);
	//performs e1 = p*e1
	MultiplyScalarWithDigitPolynomials<<<grid,block,0,aStream>>>(e1, p, elemParams->m_devModulii, elemParams->m_devMus);

	AddDigitPolynomials<<<grid,block,0,bStream>>>(c0, e0, c0, elemParams->m_devModulii);
	AddDigitPolynomials<<<grid,block,0,aStream>>>(c1, e1, c1, elemParams->m_devModulii);

	cudaStreamSynchronize(pow2Stream);
	cudaStreamDestroy(pow2Stream);
	SubtractDigitPolynomials<<<grid,block,0, bStream>>>(c0, powersOfS, c0, elemParams->m_devModulii);

	result.push_back(c0);
	result.push_back(c1);

	cudaFree(v);
	cudaFree(e0);
	cudaFree(e1);

	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);
	return std::move(result);
}

CuCRT_BGV_Ciphertext CuBGV::ReEncrypt(const std::vector<uint32_t*> &rk, const CuCRT_BGV_Ciphertext &cipher){

	auto cryptoParams = cipher.GetCryptoParameters();
	auto elemParams = cryptoParams->GetElementParams();
	auto p = cryptoParams->GetPlaintextModulus();
	auto r = cryptoParams->GetRelinWindow();
	usint nBits = elemParams->bigModulus.GetMSB();
	usint nDigits = 1;
	if (r > 0) {
		nDigits = nBits / r;
		if (nBits % r > 0){
			nDigits++;
		}
	}

	const auto& c0 = cipher.GetElementB();
	const auto& c1 = cipher.GetElementA();
	CuCRT_BGV_Ciphertext cipherNew(cryptoParams);

	//block and grid configuration for kernel launch
	usint dim = elemParams->GetRingDimension();
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(nDigits,c0.m_crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(nDigits, c0.m_crtLength);
		block = dim3(dim);
	}

	uint32_t relinPolyLength = elemParams->m_hostModulii.size()*nDigits*elemParams->GetRingDimension();

	//allocate memory to hold relin polynomial
	uint32_t *data0, *data1;
	data0 = data1 = nullptr;
	cudaError err = cudaMalloc(&data0, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for data0 polynomial\n");
	}
	err = cudaMalloc(&data1, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for data1 polynomial\n");
	}

	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);

	auto bitDecompC1 = c1.BitDecompose(r);
	MultiplyRelinPolynomials<<<grid,block,0, bStream>>>(bitDecompC1 ,rk[0] , data0, elemParams->m_devModulii, elemParams->m_devMus);
	MultiplyRelinPolynomials<<<grid,block,0, aStream>>>(bitDecompC1 ,rk[1] , data1, elemParams->m_devModulii, elemParams->m_devMus);

	//reduction code to sum and produce a shorter polynomial
	usint digiBits = std::floor(std::log2((float)nDigits)); //prev power of 2
	grid.x = 1 << digiBits;
	for(usint i= grid.x; i >0 ; i>>=1){
		grid.x = i;
		LogAddDigitPolynomials<<<grid,block,0, bStream>>>(data0, nDigits, elemParams->m_devModulii);
		LogAddDigitPolynomials<<<grid,block,0, aStream>>>(data1, nDigits, elemParams->m_devModulii);
	}

	dim3 grid2 = dim > 1024 ? dim3(1,c0.m_crtLength, dim>>10) : dim3(1,c0.m_crtLength);

	PolynomialAdditionKernel<<<grid2,block,0, bStream>>>(data0, c0.m_data, cipherNew.GetElementB().m_data, c0.m_params->m_devModulii);

	cudaMemcpyAsync(cipherNew.GetElementA().m_data, data1, c0.m_crtLength*dim*sizeof(uint32_t), cudaMemcpyDeviceToDevice, aStream);

	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);


	//release memory for data0 and data1
	cudaFree(data0);
	cudaFree(data1);

	return std::move(cipherNew);
}

__global__ void MultiplyDigitPolynomials(uint32_t *poly, uint32_t* digitPoly, uint32_t* outDigitPoly, uint32_t* modulii, uint32_t* mus){

	uint32_t polyIdx = (blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t digiIdx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[blockIdx.y + gridDim.y];
	uint32_t mod2 = modulii[blockIdx.y + 2*gridDim.y];
	uint32_t mu = mus[blockIdx.y];

	outDigitPoly[digiIdx] = ModBarretMult(poly[polyIdx],digitPoly[digiIdx], modulus, mod2, mu,bits);
}

__global__ void MultiplyScalarWithDigitPolynomials(uint32_t* digitPoly, uint32_t multVal, uint32_t* modulii, uint32_t* mus){
	uint32_t digiIdx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[blockIdx.y + gridDim.y];
	uint32_t mod2 = modulii[blockIdx.y + 2*gridDim.y];
	uint32_t mu = mus[blockIdx.y];

	digitPoly[digiIdx] = ModBarretMult(multVal,digitPoly[digiIdx], modulus, mod2, mu,bits);
}
__global__ void AddDigitPolynomials(uint32_t* digitPolyIn1, uint32_t* digitPolyIn2, uint32_t* digitPolyOut, uint32_t* modulii){
	uint32_t crtIdx = blockIdx.y;
	uint32_t digiIdx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[crtIdx];

	digitPolyOut[digiIdx] = ModAdd(digitPolyIn1[digiIdx], digitPolyIn2[digiIdx], modulus );
}
__global__ void SubtractDigitPolynomials(uint32_t* digitPolyIn1, uint32_t* digitPolyIn2, uint32_t* digitPolyOut, uint32_t* modulii){
	uint32_t digiIdx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];

	digitPolyOut[digiIdx] = ModSub(digitPolyIn1[digiIdx], digitPolyIn2[digiIdx], modulus );
}

__global__ void LogAddDigitPolynomials(uint32_t *data, uint32_t cap, uint32_t *modulii){
	uint32_t polyIdx1 = blockIdx.x;
	uint32_t polyIdx2 = 2*(gridDim.x) -1 - polyIdx1;
	if(polyIdx2>=cap)
		return;

	uint32_t idx1 = (polyIdx1*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t idx2 = (polyIdx2*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	data[idx1] = ModAdd(data[idx1], data[idx2], modulus);

/*	cudaDeviceSynchronize();

	if(gridDim.x==1){
		return;
	}

	dim3 grid(gridDim.x/2,gridDim.y);
	dim3 block(blockDim.x,1);

	//recursively launch code from first idx,i.e. idx==0
	if(idx1==0){
		LogAddDigitPolynomials<<<grid,block>>>(data, cap, modulii);
	}
	cudaDeviceSynchronize();*/
}

__global__ void PolynomialAdditionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii){

	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	outData[idx] = ModAdd(inData1[idx], inData2[idx], modulus);
}

__global__ void MultiplyRelinPolynomials(uint32_t *polyA, uint32_t* polyB, uint32_t* polyOut, uint32_t* modulii, uint32_t* mus){
	uint32_t idx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[blockIdx.y + gridDim.y];
	uint32_t mod2 = modulii[blockIdx.y + 2*gridDim.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t valA = polyA[idx];
	uint32_t valB = polyB[idx];
	polyOut[idx] = ModBarretMult(valA, valB, modulus, mod2, mu, bits);
}
