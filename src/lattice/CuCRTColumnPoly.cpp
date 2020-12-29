#include "CuCRTColumnPoly.h"

static __global__ void ReduceDiscreteUniformDistributionKernel(uint32_t *data, uint32_t *modulii);

static __global__ void ReduceGaussianDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceTernaryDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii);

static __global__ void ReduceBinaryDistributionKernel(uint32_t *data, float *dggData);

static __global__ void TimesPolyKernel(uint32_t *resultInOutColumn, uint32_t *polyData, uint32_t *modulii, uint32_t *mus);

static __global__ void TimesColumnKernel(uint32_t *resultInOutColumn, uint32_t *colmnData, uint32_t *modulii, uint32_t *mus);

static __global__ void ScalarTimesKernel(uint32_t *data, uint32_t num, uint32_t *modulii, uint32_t *mus);

static __global__ void ColumnAddKernel(uint32_t *dataInOut, uint32_t *dataIn2, uint32_t *modulii);

static __global__ void ColumnSubKernel(uint32_t *dataInOut, uint32_t *dataIn2, uint32_t *modulii);

static __global__ void LogReduce(uint32_t *data, uint32_t cap, uint32_t *modulii);


CuCRTColumnPoly::CuCRTColumnPoly(){
	this->m_crtLength = 0;
	this->m_data = nullptr;
	this->m_format = COEFFICIENT;
	this->m_params = nullptr;
	this->m_rows = 0;
}

CuCRTColumnPoly::CuCRTColumnPoly(const CuCRTColumnPoly &colmn){
	m_rows = colmn.m_rows;
	m_params = colmn.m_params;
	m_crtLength = colmn.m_crtLength;
	m_format = colmn.m_format;
	usint dim = m_params->GetRingDimension();
	uint32_t polySize = dim*m_crtLength*m_rows;
	//allocate memory
	cudaError_t err = cudaMalloc(&m_data, polySize*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Column Poly\n";
	}

	err = cudaMemcpy(m_data, colmn.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cout << "Memory copy error in copy ctor of CuCRTColumnPoly\n";
	}
}

CuCRTColumnPoly::CuCRTColumnPoly(CuCRTColumnPoly &&colmn){
	m_rows = colmn.m_rows;
	m_params = colmn.m_params;
	colmn.m_params = nullptr;
	m_crtLength = colmn.m_crtLength;
	m_format = colmn.m_format;
	m_data = colmn.m_data;
	colmn.m_data = nullptr;
}

CuCRTColumnPoly::CuCRTColumnPoly(const shared_ptr<CuCRTParams> ep, CuCRTColumnPoly::Format format, usint rows, cudaStream_t &stream){
	m_rows = rows;
	m_params = ep;
	m_crtLength = m_params->m_hostModulii.size();

	usint polySize = ep->GetRingDimension()*m_crtLength*m_rows;
	cudaError_t err = cudaMalloc(&m_data, polySize*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Column Poly\n";
		return;
	}

	//data is set to 0
	cudaMemset(m_data, 0, polySize*sizeof(uint32_t));
	m_format = format;
}

CuCRTColumnPoly::CuCRTColumnPoly(const shared_ptr<CuCRTParams> ep, CuCRTColumnPoly::Format format, CuCRTColumnPoly::NoiseType noise, usint rows, cudaStream_t &stream){
	m_rows = rows;
	m_params = ep;
	m_crtLength = ep->m_hostModulii.size();
	usint polySize = ep->GetRingDimension()*m_crtLength*m_rows;
	cudaError_t err = cudaMalloc(&m_data, polySize*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "Memory allocation error for m_data in CRT Column Poly\n";
		return;
	}

	if(noise==UNIFORM){
		this->GenerateUniformDistribution(stream);
		m_format = format;
	}
	else{
		m_format = Format::COEFFICIENT;
		switch (noise)
		{
			case NoiseType::GAUSSIAN:
				this->GenerateGaussianDistribution(stream);
				break;
			case NoiseType::TERNARY:
				this->GenerateTernaryDistribution(stream);
				break;
			case NoiseType::BINARY:
				this->GenerateBinaryDistribution(stream);
				break;
		}

		if (format != m_format) {
			SwitchFormat(stream);
		}
	}

}

CuCRTPoly CuCRTColumnPoly::operator[](uint32_t row){
	if (row >= m_rows) {
		throw std::runtime_error("row index is larger than current size");
	}

	CuCRTPoly result(m_params, CuCRTPoly::Format::EVALUATION);
	uint32_t dim = m_params->GetRingDimension();

	uint32_t *devOffset = m_data + dim*m_crtLength*row;

	cudaError_t err = cudaMemcpy(result.m_data, devOffset, dim*m_crtLength*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if(err!=cudaSuccess){
		cout << "failed in data extraction\n";
	}
	return std::move(result);
}

const CuCRTPoly CuCRTColumnPoly::operator[](uint32_t row) const{
	if (row >= m_rows) {
		throw std::runtime_error("row index is larger than current size");
	}

	CuCRTPoly result(m_params, CuCRTPoly::Format::EVALUATION);
	uint32_t dim = m_params->GetRingDimension();

	uint32_t *devOffset = m_data + dim*m_crtLength*row;

	cudaError_t err = cudaMemcpy(result.m_data, devOffset, dim*m_crtLength*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if(err!=cudaSuccess){
		cout << "failed in data extraction\n";
	}
	return std::move(result);
}


CuCRTColumnPoly& CuCRTColumnPoly::operator=(const CuCRTColumnPoly &colmn){
	m_params = colmn.m_params;
	m_crtLength = colmn.m_crtLength;
	m_format = colmn.m_format;
	m_rows = colmn.m_rows;
	usint polySize = m_params->GetRingDimension()*m_crtLength*m_rows;

	if(m_data==nullptr){
		cudaError_t err = cudaMalloc(&m_data, polySize*sizeof(uint32_t));
		if (err != cudaSuccess) {
			std::cout << "Memory allocation error in assignment operator\n";
			return *this;
		}
	}

	cudaError_t err = cudaMemcpy(m_data, colmn.m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cout << "Memory copy error in assignment operator\n";
		return *this;
	}

	return *this;
}

CuCRTColumnPoly& CuCRTColumnPoly::operator=(CuCRTColumnPoly &&colmn){
	m_params = colmn.m_params;
	colmn.m_params = nullptr;
	m_crtLength = colmn.m_crtLength;
	colmn.m_crtLength = 0;
	m_format = colmn.m_format;
	m_rows = colmn.m_rows;
	colmn.m_rows = 0;
	usint polySize = m_params->GetRingDimension()*m_crtLength*m_rows;

	cudaFree(m_data);
	m_data = colmn.m_data;
	colmn.m_data = nullptr;

	return *this;
}

CuCRTColumnPoly::~CuCRTColumnPoly(){
	cudaFree(m_data);
	m_params = nullptr;
}

void CuCRTColumnPoly::SwitchFormat(cudaStream_t &stream){
	if(m_data == nullptr){
		throw std::runtime_error("Cannot call switch format to empty values");
	}

	usint n = m_params->GetRingDimension();
	uint32_t *newValues = nullptr;
	cudaError_t e = cudaMalloc(&newValues, n*m_crtLength*m_rows*sizeof(uint32_t)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in switch format\n";
		return;
	}

	if(m_format == Format::COEFFICIENT){
		m_format = Format::EVALUATION;
		CuCRTChineseRemainderTransformFTT::ColumnForwardTransform(m_data, m_params->m_devModulii, m_params->m_devRootOfUnities, m_params->m_devMus,
				m_params->GetCyclotomicNumber(), newValues, m_crtLength, m_rows, stream);
	}
	else{
		m_format = Format::COEFFICIENT;
		CuCRTChineseRemainderTransformFTT::ColumnInverseTransform(m_data, m_params->m_devModulii, m_params->m_devInverseRootOfUnities, m_params->m_devMus,
				m_params->GetCyclotomicNumber(), newValues, m_crtLength, m_rows, stream);
	}
	cudaStreamSynchronize(stream);
	cudaFree(m_data); //release old values
	m_data = newValues;
	newValues = nullptr;
}

void CuCRTColumnPoly::GenerateUniformDistribution(cudaStream_t &stream){
	curandGenerator_t dugGen;
	curandCreateGenerator(&dugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(dugGen, time(0));
	curandSetStream(dugGen, stream);


	usint dim = m_params->GetRingDimension();
	curandStatus_t distErr = curandGenerate(dugGen, m_data, dim*m_crtLength*m_rows);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid, block;

	if(dim<=1024){
		grid = dim3(m_rows,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceDiscreteUniformDistributionKernel<<<grid,block,0,stream>>>(m_data, m_params->m_devModulii);

	curandDestroyGenerator(dugGen);
}

void CuCRTColumnPoly::GenerateGaussianDistribution(cudaStream_t &stream){

	usint dim = m_params->GetRingDimension();
	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim *m_rows* sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}

	curandGenerator_t dggGen;
	curandCreateGenerator(&dggGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(dggGen, time(0));
	curandSetStream(dggGen, stream);

	curandStatus_t distErr = curandGenerateNormal(dggGen, m_discrteGaussianDevData, dim*m_rows, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(m_rows,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(m_rows,m_crtLength,dim >> 10);
		block = dim3(1024);
	}

	ReduceGaussianDistributionKernel<<<grid,block,0,stream>>>(m_data, m_discrteGaussianDevData, m_params->m_devModulii);

	curandDestroyGenerator(dggGen);
}

void CuCRTColumnPoly::GenerateTernaryDistribution(cudaStream_t &stream){
	usint dim = m_params->GetRingDimension();

	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim *m_rows* sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}
	curandGenerator_t tugGen;
	curandCreateGenerator(&tugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(tugGen, time(0));
	curandSetStream(tugGen, stream);

	curandStatus_t distErr = curandGenerateNormal(tugGen, m_discrteGaussianDevData, dim*m_rows, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(m_rows,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceTernaryDistributionKernel<<<grid,block,0,stream>>>(m_data, m_discrteGaussianDevData, m_params->m_devModulii);

	curandDestroyGenerator(tugGen);
}

void CuCRTColumnPoly::GenerateBinaryDistribution(cudaStream_t &stream){
	usint dim = m_params->GetRingDimension();

	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, dim *m_rows* sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}
	curandGenerator_t bugGen;
	curandCreateGenerator(&bugGen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(bugGen, time(0));
	curandSetStream(bugGen, stream);

	curandStatus_t distErr = curandGenerateNormal(bugGen, m_discrteGaussianDevData, dim*m_rows, 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	dim3 grid,block;
	if(dim<=1024){
		grid = dim3(m_rows,m_crtLength);
		block = dim3(dim);
	}
	else{
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ReduceBinaryDistributionKernel<<<grid,block,0,stream>>>(m_data, m_discrteGaussianDevData);

	curandDestroyGenerator(bugGen);
}

CuCRTColumnPoly CuCRTColumnPoly::Times(const CuCRTPoly &poly) const{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	CuCRTColumnPoly result(*this);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	TimesPolyKernel<<<grid,block,0,stream>>>(result.m_data, poly.m_data, m_params->m_devModulii, m_params->m_devMus);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(result);
}

CuCRTColumnPoly CuCRTColumnPoly::Times(const CuCRTColumnPoly &colmn) const{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	CuCRTColumnPoly result(*this);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	TimesColumnKernel<<<grid,block,0,stream>>>(result.m_data, colmn.m_data, m_params->m_devModulii, m_params->m_devMus);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(result);
}

CuCRTColumnPoly CuCRTColumnPoly::Times(uint32_t num) const{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	CuCRTColumnPoly result(*this);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ScalarTimesKernel<<<grid,block,0,stream>>>(result.m_data, num, m_params->m_devModulii, m_params->m_devMus);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(result);
}

CuCRTColumnPoly CuCRTColumnPoly::Add(const CuCRTColumnPoly &colmn) const{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	CuCRTColumnPoly result(*this);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ColumnAddKernel<<<grid,block,0,stream>>>(result.m_data, colmn.m_data, m_params->m_devModulii);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(result);
}

void CuCRTColumnPoly::AddEq(const CuCRTColumnPoly &colmn){
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ColumnAddKernel<<<grid,block,0,stream>>>(this->m_data, colmn.m_data, m_params->m_devModulii);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}

CuCRTColumnPoly CuCRTColumnPoly::Subtract(const CuCRTColumnPoly &colmn) const{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	CuCRTColumnPoly result(*this);

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ColumnSubKernel<<<grid,block,0,stream>>>(result.m_data, colmn.m_data, m_params->m_devModulii);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	return std::move(result);
}

CuCRTPoly CuCRTColumnPoly::Reduce(cudaStream_t &stream) {

	usint dim = m_params->GetRingDimension();
	dim3 grid, block;
	if (dim > 1024) {
		grid = dim3(1, m_crtLength, dim >> 10);
		block = dim3(1024);
	}
	else {
		grid = dim3(1, m_crtLength);
		block = dim3(dim);
	}

	usint rowBits = std::floor(std::log2((float)m_rows)); //prev power of 2
	grid.x = 1 << rowBits;
	for(usint i= grid.x; i >0 ; i>>=1){
		grid.x = i;
		LogReduce<<<grid,block,0, stream>>>(m_data, m_rows, m_params->m_devModulii);
	}

	CuCRTPoly result(m_params, CuCRTPoly::Format::EVALUATION);
	uint32_t polySize = dim*m_crtLength;
	cudaError_t err = cudaMemcpyAsync(result.m_data, m_data, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);

	return std::move(result);
}

void CuCRTColumnPoly::SubtractEq(const CuCRTColumnPoly &colmn){
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	usint dim = m_params->GetRingDimension();

	dim3 grid,block;
	if (dim <= 1024) {
		grid = dim3(m_rows, m_crtLength);
		block = dim3(dim);
	} else {
		grid = dim3(m_rows, m_crtLength, dim >> 10);
		block = dim3(1024);
	}

	ColumnSubKernel<<<grid,block,0,stream>>>(this->m_data, colmn.m_data, m_params->m_devModulii);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}

CuCRTColumnPoly CuCRTColumnPoly::GetBitDecomposedColumnPoly(const CuCRTPoly &x, const CuCRTPoly &y, uint32_t rows, uint32_t relinWindow){
	auto ep = x.m_params;
	auto dim = ep->GetRingDimension();
	auto crtLength = ep->m_hostModulii.size();
	auto xBDptr = x.BitDecompose(relinWindow);
	auto yBDptr = y.BitDecompose(relinWindow);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	CuCRTColumnPoly result(ep, CuCRTColumnPoly::Format::EVALUATION, rows, stream);

	uint32_t polySize = dim*crtLength*(rows>>1);
	cudaError_t err = cudaMemcpyAsync(result.m_data, xBDptr, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
	if(err!=cudaSuccess){
		cout << "Error in generating column poly from bit decomposition\n";
	}
	uint32_t *offset = result.m_data + polySize;
	err = cudaMemcpyAsync(offset, yBDptr, polySize*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
	if(err!=cudaSuccess){
		cout << "Error in generating column poly from bit decomposition\n";
	}

	cudaStreamDestroy(stream);
	return std::move(result);
}


__global__ void ReduceDiscreteUniformDistributionKernel(uint32_t *data, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t number = data[idx];
	data[idx] = number % modulus;
}

__global__ void ReduceGaussianDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t dggIdx =(blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x + threadIdx.x;
	int number = llroundf(dggData[dggIdx]);
	if(number < 0){
		number *= -1;//make it positive
		data[idx] = modulus - (uint32_t)number;
	}
	else{
		data[idx] = (uint32_t)number;
	}
}

__global__ void ReduceTernaryDistributionKernel(uint32_t *data, float *dggData, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t dggIdx =(blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x + threadIdx.x;
	int number = llroundf(dggData[dggIdx]);
	if(number < 0){
		data[idx] = modulus - 1;
	}
	else if(number > 0){
		data[idx] = 1;
	}
	else{
		data[idx] = 0;
	}
}

__global__ void ReduceBinaryDistributionKernel(uint32_t *data, float *dggData){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t dggIdx =(blockIdx.z+ blockIdx.x*gridDim.z)*blockDim.x + threadIdx.x;
	int number = llroundf(dggData[dggIdx]);
	if(number < 0){
		data[idx] = 0;
	}
	else{
		data[idx] = 1;
	}
}

__global__ void TimesPolyKernel(uint32_t *resultInOutColumn, uint32_t *polyData, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t polyIdx = (blockIdx.z + blockIdx.y*gridDim.z)*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	resultInOutColumn[idx] = ModBarretMult(resultInOutColumn[idx], polyData[polyIdx], modulus, mod2, mu, bits);
}

__global__ void TimesColumnKernel(uint32_t *resultInOutColumn, uint32_t *colmnData, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	resultInOutColumn[idx] = ModBarretMult(resultInOutColumn[idx], colmnData[idx], modulus, mod2, mu, bits);
}

__global__ void ScalarTimesKernel(uint32_t *data, uint32_t num, uint32_t *modulii, uint32_t *mus){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	data[idx] = ModBarretMult(data[idx],num,modulus, mod2, mu,bits);
}

__global__ void ColumnAddKernel(uint32_t *dataInOut, uint32_t *dataIn2, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];

	dataInOut[idx] = ModAdd(dataInOut[idx], dataIn2[idx], modulus);
}

__global__ void ColumnSubKernel(uint32_t *dataInOut, uint32_t *dataIn2, uint32_t *modulii){
	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;

	uint32_t modulus = modulii[blockIdx.y];

	dataInOut[idx] = ModSub(dataInOut[idx], dataIn2[idx], modulus);
}

__global__ void LogReduce(uint32_t *data, uint32_t cap, uint32_t *modulii){
	uint32_t polyIdx1 = blockIdx.x;
	uint32_t polyIdx2 = 2*(gridDim.x) -1 - polyIdx1;
	if(polyIdx2>=cap)
		return;

	uint32_t idx1 = (polyIdx1*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t idx2 = (polyIdx2*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z)*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[blockIdx.y];

	data[idx1] = ModAdd(data[idx1], data[idx2], modulus);
}
