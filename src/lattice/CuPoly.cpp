#include "CuPoly.h"


static __global__ void GenerateUniformDistributionKernel(uint32_t *devData, uint32_t dim, CuBigInteger *mask, CuBigInteger *modulus);
static __global__ void GenerateGaussianDistributionKernel(float *dggData, uint32_t *devData, uint32_t dim, CuBigInteger *modulus);
static void printData(uint32_t* data, uint32_t dim);
static void printFloatData(float *data, uint32_t dim);

CuPoly::CuPoly(const shared_ptr<CuPolyParams> ep, CuPoly::Format format){
    usint n = ep->GetRingDimension();
    m_format = format;
	m_params = ep;

	cudaError_t error = cudaMalloc( &m_data , n*(CuBigInteger::m_nSize+1)*sizeof(uint32_t) );
	if(error!=cudaSuccess){
		printf("Memory Allocation on device failed for CuPoly class\n");
	}

	curandCreateGenerator(&m_dug, CURAND_RNG_PSEUDO_MTGP32);
	curandCreateGenerator(&m_dgg, CURAND_RNG_PSEUDO_MTGP32);
}

CuPoly::CuPoly(const shared_ptr<CuPolyParams> ep, CuPoly::Format format, CuPoly::NoiseType noise){
	usint n = ep->GetRingDimension();
	m_params = ep;
	cudaError_t error = cudaMalloc( &m_data , n*(CuBigInteger::m_nSize+1)*sizeof(uint32_t) );
	if(error!=cudaSuccess){
		printf("Memory Allocation on device failed for CuPoly class\n");
	}
	curandCreateGenerator(&m_dug, CURAND_RNG_PSEUDO_MTGP32);
	curandCreateGenerator(&m_dgg, CURAND_RNG_PSEUDO_MTGP32);

	switch (noise)
	{
	case CuPoly::NoiseType::UNIFORM:
	this->GenerateUniformDistribution();
		break;
	case CuPoly::NoiseType::GAUSSIAN:
	this->GenerateGaussianDistribution();
		break;
	}

	//since we generate everything in coeff representation
	m_format = CuPoly::Format::COEFFICIENT;
	if (format != m_format) {
		SwitchFormat();
	}
}

void CuPoly::SwitchFormat(){
	if(m_data == nullptr){
		throw std::runtime_error("Cannot call switch format to empty values");
	}
	usint n = m_params->GetRingDimension();
	uint32_t *newValues = nullptr;
	cudaError_t e = cudaMalloc(&newValues, n* (CuBigInteger::m_nSize+1)*sizeof(uint32_t)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in switch format\n";
		return;
	}

	if(m_format == CuPoly::Format::COEFFICIENT){
		m_format = CuPoly::Format::EVALUATION;
		CuChineseRemainderTransformFTT::ForwardTransform(m_data, m_params->GetModulus(), m_params->GetRootOfUnity(), m_params->GetCyclotomicNumber(), newValues );
	}
	else{
		m_format = CuPoly::Format::COEFFICIENT;
		CuChineseRemainderTransformFTT::InverseTransform(m_data, m_params->GetModulus(), m_params->GetRootOfUnity(), m_params->GetCyclotomicNumber(), newValues );
	}

	cudaFree(m_data); //release old values
	m_data = newValues;
	newValues = nullptr;
}

void CuPoly::GenerateUniformDistribution(){
	usint n = m_params->GetRingDimension();
	
	curandSetPseudoRandomGeneratorSeed(m_dug,time(0));
	curandStatus_t distErr = curandGenerate(m_dug, m_data, n*(CuBigInteger::m_nSize+1));
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerate in GenerateDiscreteUniformDistributionDevice\n";
		return;
	}

	//1024 is the max no. of threads in a block
	if (n > 1024)
	{
		GenerateUniformDistributionKernel<<<n / 1024, 1024>>>(m_data, n, m_params->m_devMask, m_params->m_devModulus);
	}
	else
	{
		GenerateUniformDistributionKernel<<<1, 1024>>>(m_data, n, m_params->m_devMask, m_params->m_devModulus);
	}
	//cudaDeviceSynchronize();
	//printData(m_data, n);
}

void CuPoly::GenerateGaussianDistribution(){
	usint n = m_params->GetRingDimension();
	curandSetPseudoRandomGeneratorSeed(m_dgg, time(0));
	cudaFree(m_discrteGaussianDevData);//erase previous data
	cudaError_t e = cudaMalloc(&m_discrteGaussianDevData, n* sizeof(float)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc in GenerateGaussianDistributionDevice\n";
		return;
	}
	curandStatus_t distErr = curandGenerateNormal(m_dgg, m_discrteGaussianDevData, n , 0, 2.0);
	if(distErr!=CURAND_STATUS_SUCCESS){
		std::cout << "error in curandGenerateNormal in GenerateGaussianDistributionDevice\n";
		return;
	}

	//1024 is the max no. of threads in a block
	if (n > 1024)
	{
		GenerateGaussianDistributionKernel<<<n/1024, 1024>>>(m_discrteGaussianDevData, m_data, n, m_params->m_devModulus);
	}
	else
	{
		GenerateGaussianDistributionKernel<<<1, 1024>>>(m_discrteGaussianDevData, m_data, n, m_params->m_devModulus);
	}
	//printData(m_data, n);
}

__global__ void GenerateUniformDistributionKernel(uint32_t *devData, uint32_t dim, CuBigInteger *mask, CuBigInteger *modulus){
	uint32_t bigintIdx = blockIdx.x * 1024 + threadIdx.x;
	uint32_t idx = bigintIdx * (CuBigInteger::m_nSize + 1);

	if (bigintIdx >= dim){
		return;
	}		

	__shared__ uint32_t maskArr[CuBigInteger::m_nSize + 1];
	if (threadIdx.x == 0)
	{
		for (uint32_t i = 0; i < CuBigInteger::m_nSize; i++)
		{
			maskArr[i] = mask->m_value[i];
		}
	}
	__syncthreads();
	
	for (uint32_t i = 0; i < CuBigInteger::m_nSize; i++)
	{
		devData[idx + i] = devData[idx + i] & maskArr[i];
	}

	CuBigInteger val(CuBigInteger::Type::EMPTY);
	
	val.m_value = devData + idx;
	val.SetMSB();
	devData[idx + CuBigInteger::m_nSize] = val.m_MSB;
	

	if (val > *modulus)
	{
		val -= *modulus;
	}
	devData[idx + CuBigInteger::m_nSize] = val.m_MSB;

	val.m_value = nullptr;
}

__global__ void GenerateGaussianDistributionKernel(float *dggData, uint32_t *devData, uint32_t dim, CuBigInteger *modulus){
	uint32_t bigintIdx = blockIdx.x * 1024 + threadIdx.x;
	uint32_t idx = bigintIdx * (CuBigInteger::m_nSize + 1);

	if (bigintIdx >= dim){
		return;
	}

	uint32_t modArr[CuBigInteger::m_nSize];
	uint32_t valArr[CuBigInteger::m_nSize];
	CuBigInteger modBigint(CuBigInteger::Type::EMPTY);
	CuBigInteger valBigint(CuBigInteger::Type::EMPTY);
	modBigint.m_value = modArr;
	valBigint.m_value = valArr;
	modBigint.m_MSB = modulus->m_MSB;
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		modBigint.m_value[i] = modulus->m_value[i];
	}
	for(uint32_t i=0; i< CuBigInteger::m_nSize-1; i++){
		valBigint.m_value[i] = 0;
	}
	valBigint.m_value[CuBigInteger::m_nSize-1] = lrintf(fabsf(dggData[bigintIdx]));
	valBigint.m_MSB = CuBigInteger::GetMSB64(valBigint.m_value[CuBigInteger::m_nSize-1]);
	if(dggData[bigintIdx] < 0 && valBigint.m_MSB!=0){
		modBigint -= valBigint;
		for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
			devData[idx+i] = modBigint.m_value[i];
		}
		devData[idx+CuBigInteger::m_nSize] = modBigint.m_MSB;
	}
	else if(dggData[bigintIdx] > 0){
		for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
			devData[idx+i] = valBigint.m_value[i];
		}
		devData[idx+CuBigInteger::m_nSize] = valBigint.m_MSB;
	}
	else{
		for(uint32_t i=0; i< (CuBigInteger::m_nSize+1); i++){
			devData[idx+i] = 0;
		}
	}

	modBigint.m_value = nullptr;
	valBigint.m_value = nullptr;
}

void printData(uint32_t *data, uint32_t dim){
	uint32_t intSize = (CuBigInteger::m_nSize+1);

	uint32_t *hostData = new uint32_t[intSize*dim];

	cudaError_t err = cudaMemcpy(hostData, data, intSize*dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "error in memory transfer\n";
	}

	for (uint32_t i = 0; i < dim; i++)
	{
		for (uint32_t j = 0; j < intSize; j++)
		{
			printf("%u \t", hostData[i * intSize + j]);
		}
	}
	printf("\n");

	delete []hostData;
}

void printFloatData(float *data, uint32_t dim){
	float *hostData = new float[dim];
	cudaMemcpy(hostData, data, dim*sizeof(float), cudaMemcpyDeviceToHost);
	for(usint i=0; i<dim;i++){
		cout << hostData[i] << '\t';
	}
	cout << '\n';
	delete []hostData;
}
