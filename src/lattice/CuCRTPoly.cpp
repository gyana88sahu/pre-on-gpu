#include "CuCRTPoly.h"

static __global__ void ReduceDiscreteUniformDistributionKernel(uint32_t *data, uint32_t *modulii);

static __global__ void ReduceDiscreteUniformDistributionKernelLarge(uint32_t *data, uint32_t *modulii);

static __global__ void ReduceGaussianDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceGaussianDistributionKernelLarge(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceTernaryDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceTernaryDistributionKernelLarge(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceBinaryDistributionKernel(uint32_t *data, float *dggData);

static __global__ void ReduceBinaryDistributionKernelLarge(uint32_t *data, float *dggData);

static __global__ void CRTInverseKernel(uint32_t *inData, uint32_t *bigPoly, CuBigInteger *bigModulus, CuBigInteger *bigMu, uint32_t *bxis, uint32_t crtSize);

//kernel function to multiply crt polynomial with value num
static __global__ void ScalarMultiplyKernel(uint32_t *inData, uint32_t *outData, uint32_t num, uint32_t *modulii, uint32_t *mus);

static __global__ void PolynomialMultiplyKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii, uint32_t *mus);

static __global__ void PolynomialAdditionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii);

static __global__ void PolynomialSubtractionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii);

static __global__ void PowersOf2Kernel(uint32_t *inData, uint32_t *outData, uint32_t* powers, uint32_t* modulii, uint32_t* mus);

static __global__ void BitDecomposeKernel(uint32_t *bigPoly, uint32_t *ans, uint32_t r);

static __global__ void SignedModKernel(uint32_t *data, uint32_t mod, uint32_t *out, CuBigInteger *bigModulus, CuBigInteger *bigMu, uint32_t *bxis, uint32_t crtLength, uint32_t qMp);


CuCRTPoly::CuCRTPoly(){
	this->m_crtLength = 0;
	this->m_data = nullptr;
	this->m_format = COEFFICIENT;
	this->m_params = nullptr;
}

CuCRTPoly::CuCRTPoly(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format){

	m_params = ep;
	m_crtLength = ep->m_hostModulii.size();
	usint crtPolyLength = ep->GetRingDimension()*m_crtLength;
	cudaError_t err = cudaMalloc(&m_data, crtPolyLength*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Poly\n";
		return;
	}
	cudaMemset(m_data, 0, crtPolyLength*sizeof(uint32_t));
	m_format = format;

}

CuCRTPoly::CuCRTPoly(const CuCRTPoly &poly){
	m_params = poly.m_params;
	m_crtLength = poly.m_crtLength;
	m_format = poly.m_format;

	usint crtPolyLength = m_params->GetRingDimension()*m_crtLength;
	cudaError_t err = cudaMalloc(&m_data, crtPolyLength*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Poly\n";
	}

	err = cudaMemcpy(m_data, poly.m_data, crtPolyLength*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cout << "Memory copy error in assignment operator\n";
	}
}

CuCRTPoly::CuCRTPoly(CuCRTPoly &&poly){
	m_params = poly.m_params;
	m_crtLength = poly.m_crtLength;
	m_format = poly.m_format;
	m_data = poly.m_data;

	poly.m_params = nullptr;
	poly.m_crtLength = 0;
	poly.m_data = nullptr;
}

CuCRTPoly& CuCRTPoly::operator=(const CuCRTPoly &poly){
	m_params = poly.m_params;
	m_crtLength = poly.m_crtLength;
	m_format = poly.m_format;
	usint crtPolyLength = m_params->GetRingDimension()*m_crtLength;

	if(m_data==nullptr){
		cudaError_t err = cudaMalloc(&m_data, crtPolyLength*sizeof(uint32_t));
		if (err != cudaSuccess) {
			std::cout << "Memory allocation error in assignment operator\n";
			return *this;
		}
	}

	cudaError_t err = cudaMemcpy(m_data, poly.m_data, crtPolyLength*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cout << "Memory copy error in assignment operator\n";
		return *this;
	}

	return *this;
}

CuCRTPoly& CuCRTPoly::operator=(CuCRTPoly &&poly){
	m_params = poly.m_params;
	m_crtLength = poly.m_crtLength;
	m_format = poly.m_format;
	poly.m_params = nullptr;
	poly.m_crtLength = 0;


	cudaFree(m_data);
	m_data = poly.m_data;
	poly.m_data = nullptr;

	return *this;
}

const CuCRTPoly& CuCRTPoly::operator=(std::initializer_list<int> rhs){
	if(m_format!=COEFFICIENT){
		throw std::runtime_error("format should be coefficient only\n");
	}
	usint dim = m_params->GetRingDimension();
	if(rhs.size()!= dim){
		throw std::runtime_error("error on polynomial lenght\n");
	}

	usint crtPolyLength = dim*m_crtLength;
	std::vector<uint32_t> hostData(crtPolyLength,0);
	for(usint i=0; i< m_crtLength; i++){
		uint32_t modulus = m_params->m_hostModulii[i];
		for(usint j=0; j<dim;j++){
			int val = *(rhs.begin()+ j);
			if(abs(val)>= modulus)
				throw runtime_error("value exceed modullus\n");

			if(val < 0){
				hostData[i*dim+j] = modulus + val;
			}
			else{
				hostData[i*dim+j] = val;
			}

		}
	}

	cudaMemcpy(m_data, hostData.data(), crtPolyLength*sizeof(uint32_t), cudaMemcpyHostToDevice);

	return *this;
}

CuCRTPoly::CuCRTPoly(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format, CuCRTPoly::NoiseType noise, cudaStream_t *stream){

	m_params = ep;
	m_crtLength = ep->m_hostModulii.size();
	usint crtPolyLength = ep->GetRingDimension()*m_crtLength;
	cudaError_t err = cudaMalloc(&m_data, crtPolyLength*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Poly\n";
		return;
	}

	if(noise==UNIFORM){
		this->GenerateUniformDistribution(stream);
		m_format = format;
	}
	else{
		m_format = CuCRTPoly::Format::COEFFICIENT;
		switch (noise)
		{
			case CuCRTPoly::NoiseType::GAUSSIAN:
				this->GenerateGaussianDistribution(stream);
				break;
			case CuCRTPoly::NoiseType::TERNARY:
				this->GenerateTernaryDistribution(stream);
				break;
			case CuCRTPoly::NoiseType::BINARY:
				this->GenerateBinaryDistribution(stream);
				break;
		}

		if (format != m_format) {
			SwitchFormat(stream);
		}
	}

}

CuCRTPoly::~CuCRTPoly(){
	cudaFree(m_data);
}

void CuCRTPoly::SwitchFormat(cudaStream_t *stream) {
	if(m_data == nullptr){
		throw std::runtime_error("Cannot call switch format to empty values");
	}

	usint n = m_params->GetRingDimension();
	uint32_t *newValues = nullptr;
	cudaError_t e = cudaMalloc(&newValues, n*m_crtLength*sizeof(uint32_t)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in switch format\n";
		return;
	}

	if(m_format == Format::COEFFICIENT){
		m_format = Format::EVALUATION;
		CuCRTChineseRemainderTransformFTT::ForwardTransform(m_data, m_params->m_devModulii, m_params->m_devRootOfUnities, m_params->m_devMus,
				m_params->GetCyclotomicNumber(), newValues, m_crtLength, stream);
	}
	else{
		m_format = Format::COEFFICIENT;
		CuCRTChineseRemainderTransformFTT::InverseTransform(m_data, m_params->m_devModulii, m_params->m_devInverseRootOfUnities, m_params->m_devMus,
				m_params->GetCyclotomicNumber(), newValues, m_crtLength, stream);
	}
	cudaStreamSynchronize(*stream);
	cudaFree(m_data); //release old values
	m_data = newValues;
	newValues = nullptr;
}

void CuCRTPoly::GenerateUniformDistribution(cudaStream_t *stream){
	curandGenerator_t dugGen;
	curandCreateGenerator(&dugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(dugGen, time(0));
	curandSetStream(dugGen, *stream);

	curandStatus_t distErr = curandGenerate(dugGen, m_data, m_crtLength*m_params->GetRingDimension());
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}
	usint dim = m_params->GetRingDimension();
	dim3 grid,block;

	if(dim<=1024){
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
		//ReduceDiscreteUniformDistributionKernel<<<grid,block>>>(m_data, m_params->m_devModulii);
	}
	else{
		grid = dim3(1, m_crtLength, dim >> 10);
		block = dim3(1024);
		//ReduceDiscreteUniformDistributionKernelLarge<<<grid,block>>>(m_data, m_params->m_devModulii);
	}

	ReduceDiscreteUniformDistributionKernel<<<grid,block, 0, *stream>>>(m_data, m_params->m_devModulii);

	curandDestroyGenerator(dugGen);
}

void CuCRTPoly::GenerateGaussianDistribution(cudaStream_t *stream){

	usint dim = m_params->GetRingDimension();

	//curandSetPseudoRandomGeneratorSeed(m_params->m_dgg, time(0));
	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim * sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}

	curandGenerator_t dggGen;
	curandCreateGenerator(&dggGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(dggGen, time(0));
	curandSetStream(dggGen, *stream);

	curandStatus_t distErr = curandGenerateNormal(dggGen, m_discrteGaussianDevData, dim, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(1, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceGaussianDistributionKernel<<<grid,block, 0, *stream>>>(m_data, m_discrteGaussianDevData, m_params->m_devModulii);

	curandDestroyGenerator(dggGen);
}

void CuCRTPoly::GenerateTernaryDistribution(cudaStream_t *stream){
	usint dim = m_params->GetRingDimension();

	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim * sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}

	curandGenerator_t tugGen;
	curandCreateGenerator(&tugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(tugGen, time(0));
	curandSetStream(tugGen, *stream);

	curandStatus_t distErr = curandGenerateNormal(tugGen, m_discrteGaussianDevData, dim, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(1, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceTernaryDistributionKernel<<<grid,block,0, *stream>>>(m_data, m_discrteGaussianDevData, m_params->m_devModulii);

	curandDestroyGenerator(tugGen);
}

void CuCRTPoly::GenerateBinaryDistribution(cudaStream_t *stream){
	usint dim = m_params->GetRingDimension();

	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim * sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}

	curandGenerator_t bugGen;
	curandCreateGenerator(&bugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(bugGen, time(0));
	curandSetStream(bugGen, *stream);

	curandStatus_t distErr = curandGenerateNormal(bugGen, m_discrteGaussianDevData, dim, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(1, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceBinaryDistributionKernel<<<grid,block, 0, *stream>>>(m_data, m_discrteGaussianDevData) ;

	curandDestroyGenerator(bugGen);
}

CuCRTPoly CuCRTPoly::Times(const CuCRTPoly &poly) const{
	if(m_format != EVALUATION || poly.m_format != EVALUATION){
		throw std::runtime_error("multiplication format not supported\n");
	}
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	CuCRTPoly ans(m_params, m_format);
	uint32_t dim = m_params->GetRingDimension();
	if(dim > 1024){
		dim3 grid(1, m_crtLength, dim >> 10);
		dim3 block(1024);
		PolynomialMultiplyKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii, m_params->m_devMus);
	}
	else{
		dim3 grid(1,m_crtLength);
		dim3 block(dim);
		PolynomialMultiplyKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii, m_params->m_devMus);
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(ans);
}

CuCRTPoly CuCRTPoly::Add(const CuCRTPoly &poly) const{
	if(m_format != EVALUATION || poly.m_format != EVALUATION){
		throw std::runtime_error("multiplication format not supported\n");
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	CuCRTPoly ans(m_params, m_format);
	uint32_t dim = m_params->GetRingDimension();
	if(dim > 1024){
		dim3 grid(1, m_crtLength, dim >> 10);
		dim3 block(1024);
		PolynomialAdditionKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii);
	}
	else{
		dim3 grid(1,m_crtLength);
		dim3 block(dim);
		PolynomialAdditionKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii);
	}
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return std::move(ans);
}

CuCRTPoly CuCRTPoly::Subtract(const CuCRTPoly &poly) const{
	if(m_format != EVALUATION || poly.m_format != EVALUATION){
		throw std::runtime_error("multiplication format not supported\n");
	}
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	CuCRTPoly ans(m_params, m_format);
	uint32_t dim = m_params->GetRingDimension();
	if(dim > 1024){
		dim3 grid(1, m_crtLength, dim >> 10);
		dim3 block(1024);
		PolynomialSubtractionKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii);
	}
	else{
		dim3 grid(1,m_crtLength);
		dim3 block(dim);
		PolynomialSubtractionKernel<<<grid, block, 0, stream>>>(m_data, poly.m_data, ans.m_data, m_params->m_devModulii);
	}
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(ans);
}

std::vector<uint32_t> CuCRTPoly::SignedMod(uint32_t mod){

	std::vector<uint32_t> host_result(m_params->GetRingDimension(),0);

	uint32_t* devResult = nullptr;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaMalloc(&devResult, m_params->GetRingDimension()*sizeof(uint32_t));
	auto qMpBI = m_params->bigModulus.Mod(BigInteger(mod));
	uint32_t dim = m_params->GetRingDimension();
	if(dim > 1024){
		SignedModKernel<<< dim >> 10,1024, 0, stream>>>(m_data, mod, devResult, m_params->m_devBigModulus, m_params->m_devBigMu, m_params->m_bxis, m_crtLength, qMpBI.ConvertToInt());
	}
	else{
		SignedModKernel<<<1,dim, 0, stream>>>(m_data, mod, devResult, m_params->m_devBigModulus, m_params->m_devBigMu, m_params->m_bxis, m_crtLength, qMpBI.ConvertToInt());
	}

	cudaMemcpyAsync(host_result.data(),devResult, m_params->GetRingDimension()*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

	cudaStreamDestroy(stream);
	return std::move(host_result);
}

void CuCRTPoly::printDataFromDevice(){
	usint dim = m_params->GetRingDimension();
	uint32_t *host = new uint32_t[m_crtLength*dim];
	cudaError_t err = cudaMemcpy(host, m_data, m_crtLength*dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if(err!= cudaSuccess){
		throw std::runtime_error("error is transferring memory from device to host in printData function\n");
	}
	for(usint i=0; i< m_crtLength; i++){
		for(usint j=0;j< dim; j++){
			std::cout << host[i*dim +j] << "  ";
		}
		std::cout << '\n' << "<------\n";
	}
	std::cout << '\n';
	delete []host;
}

uint32_t* CuCRTPoly::PowersOf2(uint32_t r, uint32_t* powersOf2, cudaStream_t *stream) const{

	if(m_format==COEFFICIENT){
		throw std::runtime_error("format needs to be in evaluation domain");
	}

	usint nBits = m_params->bigModulus.GetMSB();
	usint nDigits = 1;
	if (r > 0) {
		nDigits = nBits / r;
		if (nBits % r > 0){
			nDigits++;
		}
	}

	uint32_t relinPolyLength = m_crtLength*nDigits*m_params->GetRingDimension();

	//allocate memory for the result
	uint32_t* ans = nullptr;
	cudaError_t err = cudaMalloc(&ans, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for power of 2 polynomials\n");
	}

	uint32_t dim = m_params->GetRingDimension();
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(nDigits,m_crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(nDigits,m_crtLength);
		block = dim3(dim);
	}

	PowersOf2Kernel<<<grid,block,0,*stream>>>(m_data, ans, powersOf2, m_params->m_devModulii, m_params->m_devMus);

	return ans;
}

uint32_t* CuCRTPoly::BitDecompose(uint32_t r) const {
	uint32_t dim = m_params->GetRingDimension();
	if(m_format==COEFFICIENT){
		throw std::runtime_error("format expects polynomial to be in evaluation domain");
	}
	//converts from Evaluation to Coefficient domain
	uint32_t *coeffData = nullptr;
	cudaError_t err = cudaMalloc(&coeffData, m_crtLength*dim*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for coeffData polynomials\n");
	}

	cudaStream_t bdStream;
	cudaStreamCreate(&bdStream);

	CuCRTChineseRemainderTransformFTT::InverseTransform(m_data, m_params->m_devModulii, m_params->m_devInverseRootOfUnities, m_params->m_devMus,
					m_params->GetCyclotomicNumber(), coeffData, m_crtLength, &bdStream);

	uint32_t polySize = ceil((float)m_params->bigModulus.GetMSB()/(float)r);
	uint32_t relinPolyLength = m_crtLength*polySize*dim;

	//allocate memory for the result
	uint32_t* bdValues = nullptr;
	uint32_t* resultant = nullptr;
	err = cudaMalloc(&bdValues, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for power of 2 polynomials\n");
	}
	err = cudaMalloc(&resultant, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for power of 2 polynomials\n");
	}

	uint32_t* bigPoly = nullptr;
	cudaMalloc(&bigPoly, (CuBigInteger::m_nSize+1)*m_params->GetRingDimension()*sizeof(uint32_t));
	if(dim<=1024){
		CRTInverseKernel<<<1, dim, 0, bdStream>>>(coeffData, bigPoly, m_params->m_devBigModulus, m_params->m_devBigMu, m_params->m_bxis, m_crtLength);
		//call bit decompose kernel
		dim3 grid(polySize,m_crtLength);
		dim3 block(dim);
		BitDecomposeKernel<<<grid, block, 0, bdStream>>>(bigPoly, bdValues, r);
	}
	else{
		uint32_t gridX = dim >> 10;
		CRTInverseKernel<<<gridX, 1024, 0, bdStream>>>(coeffData, bigPoly, m_params->m_devBigModulus, m_params->m_devBigMu, m_params->m_bxis, m_crtLength);
		//call bit decompose kernel
		dim3 grid(polySize,m_crtLength, gridX);
		dim3 block(1024);
		BitDecomposeKernel<<<grid, block, 0, bdStream>>>(bigPoly, bdValues, r);
	}

	//call ntt on ans
	CuCRTChineseRemainderTransformFTT::RelinForwardTransform( bdValues, m_params->m_devModulii, m_params->m_devRootOfUnities, m_params->m_devMus,
					m_params->GetCyclotomicNumber(), resultant, m_crtLength, polySize, &bdStream);
	cudaFree(bigPoly);
	cudaFree(bdValues);
	cudaFree(coeffData);

	cudaStreamSynchronize(bdStream);
	cudaStreamDestroy(bdStream);
	return resultant;
}

CuCRTPoly CuCRTPoly::Times(uint32_t num) const{

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	CuCRTPoly ans(m_params, m_format);

	uint32_t dim = m_params->GetRingDimension();
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(1,m_crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
	}

	ScalarMultiplyKernel<<<grid,block, 0, stream>>>(m_data, ans.m_data, num, m_params->m_devModulii, m_params->m_devMus);

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(ans);
}

const CuCRTPoly& CuCRTPoly::TimesEq(uint32_t num){
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	uint32_t dim = m_params->GetRingDimension();
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(1,m_crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(1,m_crtLength);
		block = dim3(dim);
	}

	ScalarMultiplyKernel<<<grid, block, 0, stream>>>(m_data, m_data, num, m_params->m_devModulii, m_params->m_devMus);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return *this;
}

uint32_t* CuCRTPoly::GenerateRelinNoise(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format, CuCRTPoly::NoiseType noise, usint polySize, cudaStream_t *stream){

	uint32_t* result = nullptr;
	usint crtLength = ep->m_hostModulii.size();
	uint32_t dim = ep->GetRingDimension();
	uint32_t crtPolyLength = crtLength*polySize * dim;
	dim3 grid,block;
	if(dim > 1024){
		grid = dim3(polySize, crtLength,dim>>10);
		block = dim3(1024);
	}
	else{
		grid = dim3(polySize, crtLength);
		block = dim3(dim);
	}

	//allocate memory for the result
	cudaError_t err = cudaMalloc(&result, crtPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for Relin noise polynomials\n");
	}

	if(noise==UNIFORM){
		curandGenerator_t dugGen;
		curandCreateGenerator(&dugGen, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(dugGen, time(0));
		//set stream if available
		if(stream != nullptr)
			curandSetStream(dugGen, *stream);

		curandStatus_t distErr = curandGenerate(dugGen, result, crtPolyLength);
		if (distErr != CURAND_STATUS_SUCCESS) {
			std::cout << "error in curandGenerate in GenerateRelinNoise\n";
			return nullptr;
		}
		ReduceDiscreteUniformDistributionKernel<<<grid,block,0,*stream>>>(result, ep->m_devModulii);
		curandDestroyGenerator(dugGen);

		return result;
	}
	else{//noise generation for binary, ternary and gaussian
		float* dggData = nullptr;
		usint dim = polySize*ep->GetRingDimension();
		err = cudaMalloc(&dggData, dim*sizeof(float));
		if(err!=cudaSuccess){
			throw std::runtime_error("error in allocating dgg memory for Relin noise polynomials\n");
		}
		curandGenerator_t dggGen;
		curandCreateGenerator(&dggGen, CURAND_RNG_PSEUDO_MTGP32);
		curandSetPseudoRandomGeneratorSeed(dggGen, time(0));

		//set stream if available
		if(stream != nullptr)
			curandSetStream(dggGen, *stream);

		curandStatus_t distErr = curandGenerateNormal(dggGen, dggData, dim, 0, 2.0);
		if(distErr!=CURAND_STATUS_SUCCESS){
			std::cout << "error in curandGenerate in GenerateRelinNoise\n";
			return nullptr;
		}

		if(noise==GAUSSIAN){
			ReduceGaussianDistributionKernel<<<grid,block,0,*stream>>>(result, dggData, ep->m_devModulii);
		}
		else if(noise==BINARY){
			ReduceBinaryDistributionKernel<<<grid,block,0,*stream>>>(result, dggData);
		}
		else if(noise==TERNARY){
			ReduceTernaryDistributionKernel<<<grid,block,0,*stream>>>(result, dggData, ep->m_devModulii);
		}
		cudaFree(dggData);
		curandDestroyGenerator(dggGen);
	}

	if(format!=COEFFICIENT){
		//call ntt
		uint32_t* resultOut = nullptr;
		cudaError_t err = cudaMalloc(&resultOut, crtPolyLength*sizeof(uint32_t));
		if(err!=cudaSuccess){
			throw std::runtime_error("error in allocating memory for Relin noise polynomials\n");
		}
		CuCRTChineseRemainderTransformFTT::RelinForwardTransform(result, ep->m_devModulii,ep->m_devRootOfUnities, ep->m_devMus, ep->GetCyclotomicNumber(),
				resultOut, ep->m_hostModulii.size(), polySize, stream);
		cudaStreamSynchronize(*stream);
		cudaFree(result);
		result = resultOut;
		resultOut = nullptr;
	}

	return result;

}

std::vector<NativePoly> CuCRTPoly::ConvertToNativePoly() const{

	std::vector<NativePoly> result;

	usint dim = m_params->GetRingDimension();
	usint dataSize = m_crtLength*dim;
	std::vector<uint32_t> hostData(dataSize,0);
	cudaMemcpy(hostData.data(), m_data, dataSize*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	return std::move(result);
}

std::vector<uint32_t> CuCRTPoly::ConvertRelinPolyToVector(uint32_t *data, shared_ptr<CuCRTParams> ep, usint r){
	usint nBits = ep->bigModulus.GetMSB();
	usint nDigits = 1;
	if (r > 0) {
		nDigits = nBits / r;
		if (nBits % r > 0){
			nDigits++;
		}
	}
	uint32_t relinPolyLength = ep->m_hostModulii.size()*nDigits*ep->GetRingDimension();
	std::vector<uint32_t> result(relinPolyLength, 0);
	cudaMemcpy(result.data(), data, relinPolyLength*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return std::move(result);
}

__global__ void ReduceDiscreteUniformDistributionKernel(uint32_t *data, uint32_t *modulii){
	uint32_t idx = (blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t number = data[idx];
	data[idx] = number % modulus;
}

__global__ void ReduceDiscreteUniformDistributionKernelLarge(uint32_t *data, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t number = data[idx];
	data[idx] = number % modulus;
}


__global__ void ReduceGaussianDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t dggIdx =(blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x + threadIdx.x;
	float number = dggData[dggIdx];
	if(number < 0){
		data[idx] = modulus - (uint32_t)llroundf(fabsf(number));
	}
	else{
		data[idx] = (uint32_t)llroundf(number);
	}
}

__global__ void ReduceGaussianDistributionKernelLarge(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t dggIdx =(blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x + threadIdx.x;
	float number = dggData[dggIdx];
	if(number < 0){
		data[idx] = modulus - (uint32_t)llroundf(fabsf(number));
	}
	else{
		data[idx] = (uint32_t)llroundf(number);
	}
}


__global__ void ReduceTernaryDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t dggIdx = (blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x +threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	float val = dggData[dggIdx];
	uint32_t roundedVal = llroundf(fabsf(val));

	if(roundedVal == 0){
		data[idx] = 0;
	}
	else if (val >0){
		data[idx] = 1;
	}
	else{
		data[idx] = modulus -1;
	}
}

__global__ void ReduceTernaryDistributionKernelLarge(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t dggIdx = (blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x +threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	float val = dggData[dggIdx];
	uint32_t roundedVal = llroundf(fabsf(val));

	if(roundedVal == 0){
		data[idx] = 0;
	}
	else if (val >0){
		data[idx] = 1;
	}
	else{
		data[idx] = modulus -1;
	}
}

__global__ void ReduceBinaryDistributionKernel(uint32_t *data, float *dggData){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t dggIdx = (blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x +threadIdx.x;
	float number = dggData[dggIdx];
	if(number <= 0){
		data[idx] = 0;
	}
	else {
		data[idx] = 1;
	}

}

__global__ void ReduceBinaryDistributionKernelLarge(uint32_t *data, float *dggData){

	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t dggIdx = (blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x +threadIdx.x;
	float number = dggData[dggIdx];
	if(number <= 0){
		data[idx] = 0;
	}
	else {
		data[idx] = 1;
	}

}

__global__ void ScalarMultiplyKernel(uint32_t *inData, uint32_t *outData, uint32_t num, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[gridDim.y + blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y + blockIdx.y];
	uint32_t mu = mus[blockIdx.y];

	outData[idx] = ModBarretMult( inData[idx], num, modulus, mod2, mu, bits);
}

__global__ void PolynomialMultiplyKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii, uint32_t *mus){

	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[gridDim.y + blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y + blockIdx.y];
	uint32_t mu = mus[blockIdx.y];

	outData[idx] = ModBarretMult( inData1[idx], inData2[idx], modulus, mod2, mu, bits);
}

__global__ void PolynomialAdditionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	outData[idx] = ModAdd(inData1[idx], inData2[idx], modulus);
}

__global__ void PolynomialSubtractionKernel(uint32_t *inData1, uint32_t *inData2, uint32_t *outData, uint32_t *modulii){
	uint32_t blockId = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = blockId*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	outData[idx] = ModSub(inData1[idx], inData2[idx], modulus);
}

__global__ void PowersOf2Kernel(uint32_t *inData, uint32_t *outData, uint32_t* powers, uint32_t* modulii, uint32_t* mus){

	uint32_t block = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	uint32_t inIdx = (blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t outIdx = (block)*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t bits = modulii[blockIdx.y + gridDim.y];
	uint32_t mod2 = modulii[blockIdx.y + 2*gridDim.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t powerValue = powers[blockIdx.x*gridDim.y + blockIdx.y];
	outData[outIdx] = ModBarretMult(inData[inIdx],powerValue,modulus, mod2, mu,bits);
}

__global__ void CRTInverseKernel(uint32_t *inData, uint32_t *bigPoly, CuBigInteger *bigModulus, CuBigInteger *bigMu, uint32_t *bxis, uint32_t crtSize){

	uint32_t memory[3*(CuBigInteger::m_nSize+1)];

	CuBigInteger bigCoeff(CuBigInteger::Type::EMPTY);
	bigCoeff.m_value = memory;

	CuBigInteger bxi(CuBigInteger::Type::EMPTY);
	bxi.m_value = memory + (CuBigInteger::m_nSize+1);

	CuBigInteger mult(CuBigInteger::Type::EMPTY);
	mult.m_value = memory + 2*(CuBigInteger::m_nSize+1);

	for(int i=0; i< CuBigInteger::m_nSize; i++){
		bigCoeff.m_value[i] = 0;
	}
	bigCoeff.m_MSB =0;

	uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t n = gridDim.x*blockDim.x;

	for(int i=0;i<crtSize;i++){
		uint32_t coeff = inData[i*n + idx];
		for(int j=0; j< CuBigInteger::m_nSize; j++){
			bxi.m_value[j] = bxis[i*(CuBigInteger::m_nSize+1) + j];
		}
		bxi.m_MSB = bxis[(i+1)*(CuBigInteger::m_nSize+1)-1];
		bxi.MulByUintToInt(coeff, &mult);
		bigCoeff += mult;
	}

	bigCoeff.ModBarrettInPlace(*bigModulus, *bigMu);

	for(int i=0; i< CuBigInteger::m_nSize; i++){
		bigPoly[idx*(CuBigInteger::m_nSize+1) + i] = bigCoeff.m_value[i];
	}
	bigPoly[(idx+1)*(CuBigInteger::m_nSize+1) -1] = bigCoeff.m_MSB;

	bigCoeff.m_value = nullptr;
	bxi.m_value = nullptr;
	mult.m_value = nullptr;
}

__global__ void BitDecomposeKernel(uint32_t *bigPoly, uint32_t *ans, uint32_t r){
	uint32_t inIdx = blockIdx.z*blockDim.x + threadIdx.x;
	uint32_t block = gridDim.y*gridDim.z*blockIdx.x + blockIdx.y*gridDim.z + blockIdx.z;
	uint32_t outIdx = block*blockDim.x + threadIdx.x;
	CuBigInteger coeff(CuBigInteger::Type::EMPTY);
	coeff.m_MSB = bigPoly[(inIdx+1)*(CuBigInteger::m_nSize+1)-1];
	coeff.m_value = bigPoly + inIdx*(CuBigInteger::m_nSize+1);
	ans[outIdx] = coeff.GetDigitAtIndexForBase(blockIdx.x + 1, r);
	coeff.m_value = nullptr;
}

__global__ void SignedModKernel(uint32_t *data, uint32_t mod, uint32_t *out, CuBigInteger *bigModulus, CuBigInteger *bigMu, uint32_t *bxis, uint32_t crtLength, uint32_t qMp){

	uint32_t n = blockDim.x*gridDim.x;
	uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	uint32_t memory[4*(CuBigInteger::m_nSize+1)];

	CuBigInteger bigCoeff(CuBigInteger::Type::EMPTY);
	bigCoeff.m_value = memory;

	CuBigInteger bxi(CuBigInteger::Type::EMPTY);
	bxi.m_value = memory + (CuBigInteger::m_nSize+1);

	CuBigInteger mult(CuBigInteger::Type::EMPTY);
	mult.m_value = memory + 2*(CuBigInteger::m_nSize+1);

	CuBigInteger halfQ(CuBigInteger::Type::EMPTY);
	halfQ.m_value = memory + 3*(CuBigInteger::m_nSize+1);
	halfQ = *bigModulus;
	halfQ >>=1;
	for(int i=0; i< CuBigInteger::m_nSize; i++){
		bigCoeff.m_value[i] = 0;
	}
	bigCoeff.m_MSB = 0;

	for(int i=0;i<crtLength;i++){
		uint32_t coeff = data[i*n + idx];
		for(int j=0; j< CuBigInteger::m_nSize; j++){
			bxi.m_value[j] = bxis[i*(CuBigInteger::m_nSize+1) + j];
		}
		bxi.m_MSB = bxis[(i+1)*(CuBigInteger::m_nSize+1)-1];
		bxi.MulByUintToInt(coeff, &mult);
		bigCoeff += mult;
	}

	for(int i=0; i< CuBigInteger::m_nSize; i++){
		mult.m_value[i] = 0;
	}
	mult.m_value[CuBigInteger::m_nSize-1] = mod;
	mult.m_MSB = CuBigInteger::GetMSB64(mod);

	bigCoeff.ModBarrettInPlace(*bigModulus, *bigMu);
	uint32_t ans = 0;
	if (bigCoeff > halfQ) {
		bigCoeff.ModSelf(mult);
		uint32_t a = bigCoeff.m_value[CuBigInteger::m_nSize-1];
		if(a>=qMp){
			ans = a-qMp;
		}
		else{
			ans = a + mod;
			ans -= qMp;
		}
	}
	else {
	  bigCoeff.ModSelf(mult);
	  ans = bigCoeff.m_value[CuBigInteger::m_nSize-1];
	}

	out[idx] = ans;

	//release memory
	bigCoeff.m_value = nullptr;
	bxi.m_value = nullptr;
	mult.m_value = nullptr;
	halfQ.m_value = nullptr;
}

