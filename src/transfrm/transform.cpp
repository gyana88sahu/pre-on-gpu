#include "transform.h"

/*NTT functions for Big Integers
*/
//PreCompute for Big Integers: Builds roots of unity and inverse root of unity tables.
static __global__ void PreComputeRoot(uint32_t *table, CuBigInteger *modulus, CuBigInteger *root, CuBigInteger *mu, uint32_t n);
static __global__ void CRTPreComputeRoot(uint32_t *table, uint32_t *modulii, uint32_t *roots,uint32_t *mus, usint n);
static __global__ void PrepareInput(uint32_t *inData, uint32_t *m_rTableDev, CuBigInteger *devModulus, CuBigInteger *devMu, uint32_t *outData, uint32_t *bitReverseLookup,uint32_t n);
static __global__ void PrepareCRTInput(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData, uint32_t *bitRevTable, uint32_t n);
static __global__ void Butterfly(uint32_t *data, uint32_t jump, uint32_t n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu);

static __global__ void Butterfly2(uint32_t *data, uint32_t stage, uint32_t log2n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu);
static __global__ void Butterfly3(uint32_t *data, uint32_t log2n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu);
static __global__ void CRTButterfly(uint32_t *data, uint32_t log2n, uint32_t *rTable, uint32_t *modulii, uint32_t *mus);

//new methods to tie up ntt and crt
//butterfly does the reversal first and then starts the butterfly process
static __global__ void CRTButterflyNew(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *bitRevTable, uint32_t *outData);

static __global__ void BitReverse(uint32_t *inData, uint32_t *bitRevTable, uint32_t *outData);
static __global__ void RelinBitReverse(uint32_t *inData, uint32_t *bitRevTable, uint32_t *outData);


//computes the NTT of large polynomials upto 10 stages
static __global__ void CRTButterflyNewLargeStage10(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus);

static __global__ void CRTButterflyNewLarge(uint32_t *inData, uint32_t log2n, uint32_t stage, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData);

static __global__ void RelinCRTButterflyLarge(uint32_t *inData, uint32_t log2n, uint32_t stage, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData);

//prepare just multiplies with the twiddle factors
static __global__ void PrepareCRTInputNew(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus);

static __global__ void PrepareCRTInputNewLarge(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus);

static __global__ void ScalarMultiply(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus);

static __global__ void ScalarMultiplyLarge(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus);

static __global__ void ScalarMultiplyColumn(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus);

static __global__ void RelinPrepareCRTInput(uint32_t *data, uint32_t *table, uint32_t *modulii,uint32_t *mus);

static __global__ void RelinCRTButterfly(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *bitRevTable, uint32_t *outData);

static __global__ void RelinCRTButterflyLargeStage10(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus);

//Utility functions to print data for device memory
static void printData(uint32_t *data, uint32_t dim);
void printCRTData(uint32_t *data, uint32_t dim, uint32_t crtSize);

//static tables for storing root and inverse root of unity tables.
uint32_t* CuChineseRemainderTransformFTT::m_rTableDev = nullptr;
uint32_t* CuChineseRemainderTransformFTT::m_rInverseTableDev = nullptr;
CuBigInteger* CuChineseRemainderTransformFTT::devModulus = nullptr;
CuBigInteger* CuChineseRemainderTransformFTT::devRoot = nullptr;
CuBigInteger* CuChineseRemainderTransformFTT::devRootInverse = nullptr;
CuBigInteger* CuChineseRemainderTransformFTT::devMu = nullptr;
uint32_t* CuChineseRemainderTransformFTT::bitReverseTable = nullptr;

uint32_t* CuCRTChineseRemainderTransformFTT::m_rTableDev = nullptr;
uint32_t* CuCRTChineseRemainderTransformFTT::m_rInverseTableDev = nullptr;
uint32_t* CuCRTChineseRemainderTransformFTT::bitReverseTable = nullptr;
uint32_t* CuCRTChineseRemainderTransformFTT::m_nInverseTable = nullptr;

void CuChineseRemainderTransformFTT::ForwardTransform(uint32_t *inData, const BigInteger &modulus, const BigInteger &root, usint cycloOrder, uint32_t *outData){
	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	if (root == BigInteger(0) || root == BigInteger(1)) {
		throw std::runtime_error("root cannot be 1 or 0\n");
	}

	usint n = cycloOrder/2;

	//compute table if empty
	if(m_rTableDev == nullptr){
		if(devModulus==nullptr){
			CuBigInteger::InitializeDeviceVariables(&devModulus, modulus.ToString());
			CuBigInteger::InitializeDeviceVariables(&devRoot, root.ToString());
			auto mu = ComputeMu(modulus);
			CuBigInteger::InitializeDeviceVariables(&devMu, mu.ToString());
		}

		if (bitReverseTable == nullptr) {
			CuChineseRemainderTransformFTT::ComputeBitReverseTable(n);
		}
		cudaError_t e = cudaMalloc(&m_rTableDev, n* (CuBigInteger::m_nSize+1)*sizeof(uint32_t)); //allocate memory
		if(e!=cudaSuccess){
			std::cout << "error in cudaMalloc for rTableDev\n";
			return;
		}
		PreComputeRoot<<<1,1>>>(m_rTableDev, devModulus, devRoot, devMu, n); //cannot launch threads coz we don't have ModExp in CuBigInteger
		cudaDeviceSynchronize();
	}

	//force the input to be fixed input for debugging purposes
	uint32_t *cpuInData = new uint32_t[n*(CuBigInteger::m_nSize+1)]{0};
	std::vector<usint> inputValues = {1,2,3,4,5,6,7,8};
	for(usint i=0; i<8;i++){
		BigInteger val(inputValues[i]);
		cpuInData[i*(CuBigInteger::m_nSize+1)+2] = inputValues[i];
		cpuInData[i*(CuBigInteger::m_nSize+1)+3] = val.GetMSB();
	}
	cudaMemcpy(inData, cpuInData, n*(CuBigInteger::m_nSize+1)*sizeof(uint32_t),cudaMemcpyHostToDevice);

	delete []cpuInData;

	/*std::cout << "before input data is processed\n";
	printData(inData,n);*/

	//do multiplication by w_i's for FTT optimization and scramble for DIT Cooley-Tukey
	if(n>1024){
		PrepareInput<<<n/1024,1024>>>(inData, m_rTableDev, devModulus, devMu, outData, bitReverseTable, n);
	}
	else{
		PrepareInput<<<1,n>>>(inData, m_rTableDev, devModulus, devMu, outData, bitReverseTable, n);
	}
	//printData(outData,n);
	//CuNumberTheoreticTransform::ForwardTransform(outData, m_rTableDev, n);
	printData(outData,n);
}

void CuChineseRemainderTransformFTT::InverseTransform(uint32_t *inData, const BigInteger &modulus, const BigInteger &root, usint cycloOrder, uint32_t *outData ){

}

void CuChineseRemainderTransformFTT::ComputeBitReverseTable(usint n){

	uint32_t *cpuBitReverseTable = new uint32_t[n];
	usint msb = GetMSB64(n-1);
	for(usint i=0; i< n; i++){
		cpuBitReverseTable[i] = ReverseBits(i,msb);
	}

	cudaError_t e = cudaMalloc(&bitReverseTable, n*sizeof(uint32_t)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc for bitReverseTable\n";
		return;
	}
	e = cudaMemcpy(bitReverseTable, cpuBitReverseTable, n*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(e!=cudaSuccess){
		std::cout << "error in cudaMemcpy for bitReverseTable\n";
		return;
	}
	delete []cpuBitReverseTable;
}

void CuCRTChineseRemainderTransformFTT::ForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, cudaStream_t *stream){
	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	usint n = cycloOrder/2;

	//compute table if empty
	if(m_rTableDev == nullptr){

		if (bitReverseTable == nullptr) {
			CuCRTChineseRemainderTransformFTT::ComputeBitReverseTable(n);
		}
		cudaError_t e = cudaMalloc(&m_rTableDev, n* towers*sizeof(uint32_t)); //allocate memory

		if(e!=cudaSuccess){
			std::cout << "error in cudaMalloc for rTableDev\n";
			return;
		}

		CRTPreComputeRoot<<<towers,1,0, *stream>>>(m_rTableDev, modulii, roots, mus, n); //cannot launch threads coz we don't have ModExp
		cudaStreamSynchronize(*stream);
	}

	//force the input to be fixed input for debugging purposes
	/*uint32_t *cpuInData = new uint32_t[n*towers]{0};
	std::vector<usint> inputValues = {1,2,3,4,5,6,7,8};
	for(usint i=0; i<towers;i++){
		for(usint j=0; j<8; j++){
			cpuInData[n*i+j] = inputValues[j];
		}
	}
	cudaMemcpy(inData, cpuInData, n*towers*sizeof(uint32_t),cudaMemcpyHostToDevice);

	delete []cpuInData;*/

	//prepareInput
	//do multiplication by w_i's for FTT optimization and scramble for DIT Cooley-Tukey
	uint32_t log2n = log2((float)n);
	if (n > 1024) {
		dim3 grid;
		grid.y = n >> 10;
		grid.x = towers;
		PrepareCRTInputNewLarge<<<grid,1024, 0, *stream>>>(inData, m_rTableDev, modulii, mus);

		BitReverse<<<grid, 1024, 0, *stream>>>(inData, bitReverseTable, outData);

		CRTButterflyNewLargeStage10<<<grid, 1024, 0, *stream>>>(outData, log2n, m_rTableDev, modulii, mus);

		for(usint i= 11; i <= log2n; i++){
			if(i%2==0)
				CRTButterflyNewLarge<<<grid, 1024, 0, *stream>>>(inData, log2n, i, m_rTableDev, modulii, mus, outData);
			else
				CRTButterflyNewLarge<<<grid, 1024, 0, *stream>>>(outData, log2n, i, m_rTableDev, modulii, mus, inData);
		}

		if(log2n%2!=0){
			cudaMemcpyAsync(outData, inData, n*towers*sizeof(uint32_t), cudaMemcpyDeviceToDevice, *stream);
		}

	}
	else{
		//PrepareCRTInput<<<towers, n>>>(inData, m_rTableDev, modulii, mus, outData, bitReverseTable, n);
		PrepareCRTInputNew<<<towers,n,0, *stream>>>(inData, m_rTableDev, modulii, mus);

		//CRTButterfly<<<towers, n>>>(outData, log2n, m_rTableDev, modulii, mus);

		CRTButterflyNew<<<towers,n,0, *stream>>>(inData, log2n, m_rTableDev, modulii, mus, bitReverseTable, outData);
	}

	cudaStreamSynchronize(*stream);

	//printCRTData(outData, n, 3);

}

void CuCRTChineseRemainderTransformFTT::InverseTransform(uint32_t *inData, uint32_t *modulii, uint32_t *rootIs, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, cudaStream_t *stream){

	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	usint n = cycloOrder/2;

	//compute table if empty
	if(m_rInverseTableDev == nullptr){

		if (bitReverseTable == nullptr) {
			CuCRTChineseRemainderTransformFTT::ComputeBitReverseTable(n);
		}
		cudaError_t e = cudaMalloc(&m_rInverseTableDev, n* towers*sizeof(uint32_t)); //allocate memory

		if(e!=cudaSuccess){
			std::cout << "error in cudaMalloc for rInverseTableDev\n";
			return;
		}

		CRTPreComputeRoot<<<towers,1, 0, *stream>>>(m_rInverseTableDev, modulii, rootIs, mus, n); //cannot launch threads coz we don't have ModExp
		cudaStreamSynchronize(*stream);

		cudaMalloc(&m_nInverseTable, towers*sizeof(uint32_t));
		uint32_t *hostModulii, *hostNInverses;
		cudaMallocHost((void**)&hostModulii, towers*sizeof(uint32_t));
		cudaMallocHost((void**)&hostNInverses, towers*sizeof(uint32_t));
		cudaMemcpy(hostModulii, modulii, towers*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for(usint i=0; i<towers;i++){
			NativeInteger num(n);
			num = num.ModInverse(NativeInteger(hostModulii[i]));
			hostNInverses[i] = num.ConvertToInt();
		}
		cudaMemcpy(m_nInverseTable, hostNInverses, towers*sizeof(uint32_t), cudaMemcpyHostToDevice);

		cudaFreeHost(hostModulii);
		cudaFreeHost(hostNInverses);
	}

	uint32_t log2n = log2((float)n);
	if (n > 1024) {
		dim3 grid;
		grid.y = n >> 10;
		grid.x = towers;

		BitReverse<<<grid, 1024, 0, *stream>>>(inData, bitReverseTable, outData);

		CRTButterflyNewLargeStage10<<<grid, 1024, 0, *stream>>>(outData, log2n, m_rInverseTableDev, modulii, mus);

		for(usint i= 11; i <= log2n; i++){
			if(i%2==0)
				CRTButterflyNewLarge<<<grid, 1024, 0, *stream>>>(inData, log2n, i, m_rInverseTableDev, modulii, mus, outData);
			else
				CRTButterflyNewLarge<<<grid, 1024, 0, *stream>>>(outData, log2n, i, m_rInverseTableDev, modulii, mus, inData);
		}
		if(log2n%2!=0){
			cudaMemcpyAsync(outData, inData, n*towers*sizeof(uint32_t), cudaMemcpyDeviceToDevice, *stream);
		}

		ScalarMultiplyLarge<<<grid, 1024, 0, *stream>>>(outData, m_nInverseTable, modulii, mus);

		PrepareCRTInputNewLarge<<<grid, 1024, 0, *stream>>>(outData, m_rInverseTableDev, modulii, mus);
	}
	else{
		CRTButterflyNew<<<towers,n, 0, *stream>>>(inData,log2n, m_rInverseTableDev, modulii, mus, bitReverseTable, outData);
		//multiply with n inverse modulus
		ScalarMultiply<<<towers,n, 0, *stream>>>(outData, m_nInverseTable, modulii, mus);
		//twiddle multiplication
		PrepareCRTInputNew<<<towers,n, 0, *stream>>>(outData, m_rInverseTableDev, modulii, mus);
	}

	cudaStreamSynchronize(*stream);
	//printCRTData(outData, n, 3);

}

void CuCRTChineseRemainderTransformFTT::ComputeBitReverseTable(usint n){
	uint32_t *cpuBitReverseTable = nullptr;
	cudaMallocHost((void**)&cpuBitReverseTable, n*sizeof(uint32_t));
	usint msb = GetMSB64(n-1);
	for(usint i=0; i< n; i++){
		cpuBitReverseTable[i] = ReverseBits(i,msb);
	}

	cudaError_t e = cudaMalloc(&bitReverseTable, n*sizeof(uint32_t)); //allocate memory
	if(e!=cudaSuccess){
		std::cout << "error in cudaMalloc for bitReverseTable\n";
		return;
	}
	e = cudaMemcpy(bitReverseTable, cpuBitReverseTable, n*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(e!=cudaSuccess){
		std::cout << "error in cudaMemcpy for bitReverseTable\n";
		return;
	}
	cudaFreeHost(cpuBitReverseTable);
}

void CuCRTChineseRemainderTransformFTT::RelinForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint polySize, cudaStream_t *stream){

	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	usint n = cycloOrder/2;

	//assume all the tables are precomputed before this operation

	uint32_t log2n = log2((float)n);
	if(n>1024){
		uint32_t gridZ = n >> 10;
		dim3 grid(polySize, towers, gridZ );
		dim3 block(1024);

		RelinPrepareCRTInput<<<grid, block,0, *stream>>>(inData, m_rTableDev, modulii, mus);
		RelinBitReverse<<<grid, block,0,*stream>>>(inData, bitReverseTable, outData);
		RelinCRTButterflyLargeStage10<<<grid, block,0,*stream>>>(outData, log2n, m_rTableDev, modulii, mus);
		for(usint i= 11; i <= log2n; i++){
			if(i%2==0)
				RelinCRTButterflyLarge<<<grid, block, 0, *stream>>>(inData, log2n, i, m_rTableDev, modulii, mus, outData);
			else
				RelinCRTButterflyLarge<<<grid, block, 0, *stream>>>(outData, log2n, i, m_rTableDev, modulii, mus, inData);
		}

		if(log2n%2!=0){
			cudaMemcpyAsync(outData, inData, polySize*n*towers*sizeof(uint32_t), cudaMemcpyDeviceToDevice, *stream);
		}

	}
	else{
		dim3 grid(polySize, towers);
		dim3 block(n,1);
		RelinPrepareCRTInput<<<grid, block,0, *stream>>>(inData, m_rTableDev, modulii, mus);
		RelinCRTButterfly<<<grid, block,0, *stream>>>(inData, log2n, m_rTableDev, modulii, mus, bitReverseTable, outData);
	}

	cudaStreamSynchronize(*stream);
}

void CuCRTChineseRemainderTransformFTT::ColumnForwardTransform(uint32_t *inData, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint rows, cudaStream_t &stream){

	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	usint n = cycloOrder/2;

	//assume all the tables are precomputed before this operation

	uint32_t log2n = log2((float)n);
	if(n>1024){
		uint32_t gridZ = n >> 10;
		dim3 grid(rows, towers, gridZ );
		dim3 block(1024);

		RelinPrepareCRTInput<<<grid, block,0,stream>>>(inData, m_rTableDev, modulii, mus);
		RelinBitReverse<<<grid, block,0,stream>>>(inData, bitReverseTable, outData);
		RelinCRTButterflyLargeStage10<<<grid, block,0,stream>>>(outData, log2n, m_rTableDev, modulii, mus);
		for(usint i= 11; i <= log2n; i++){
			if(i%2==0)
				RelinCRTButterflyLarge<<<grid, block, 0,stream>>>(inData, log2n, i, m_rTableDev, modulii, mus, outData);
			else
				RelinCRTButterflyLarge<<<grid, block, 0,stream>>>(outData, log2n, i, m_rTableDev, modulii, mus, inData);
		}

		if(log2n%2!=0){
			cudaMemcpyAsync(outData, inData, n*towers*rows*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
		}

	}
	else{
		dim3 grid(rows, towers);
		dim3 block(n);
		RelinPrepareCRTInput<<<grid, block,0,stream>>>(inData, m_rTableDev, modulii, mus);
		RelinCRTButterfly<<<grid, block,0,stream>>>(inData, log2n, m_rTableDev, modulii, mus, bitReverseTable, outData);
	}

	cudaStreamSynchronize(stream);
}

void CuCRTChineseRemainderTransformFTT::ColumnInverseTransform(uint32_t *inData, uint32_t *modulii, uint32_t *rootIs, uint32_t *mus, usint cycloOrder, uint32_t *outData, usint towers, usint rows, cudaStream_t &stream){
	if (inData == nullptr || outData == nullptr) {
		throw std::runtime_error("input or output pointer is empty\n");
	}

	usint n = cycloOrder/2;

	//assume all the tables are precomputed before this operation

	uint32_t log2n = log2((float)n);
	if(n>1024){
		uint32_t gridZ = n >> 10;
		dim3 grid(rows, towers, gridZ );
		dim3 block(1024);

		//bit reverse
		RelinBitReverse<<<grid, block,0,stream>>>(inData, bitReverseTable, outData);

		//10 stage fft
		RelinCRTButterflyLargeStage10<<<grid,block,0,stream>>>(outData, log2n, m_rInverseTableDev, modulii, mus);

		//rest of stages of fft
		for(usint i= 11; i <= log2n; i++){
			if(i%2==0)
				RelinCRTButterflyLarge<<<grid,block,0,stream>>>(inData, log2n, i, m_rInverseTableDev, modulii, mus, outData);
			else
				RelinCRTButterflyLarge<<<grid,block,0,stream>>>(outData, log2n, i, m_rInverseTableDev, modulii, mus, inData);
		}

		if(log2n%2!=0){
			cudaMemcpyAsync(outData, inData, n*towers*rows*sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
		}


		//scalar multiplication with nInverse
		ScalarMultiplyColumn<<<grid,block,0,stream>>>(outData,m_nInverseTable,modulii,mus);

		//multiplication with inverse table
		RelinPrepareCRTInput<<<grid, block,0,stream>>>(outData, m_rInverseTableDev, modulii, mus);
	}
	else{
		dim3 grid(rows, towers);
		dim3 block(n);

		RelinCRTButterfly<<<grid,block,0,stream>>>(inData, log2n, m_rTableDev, modulii, mus, bitReverseTable, outData);

		ScalarMultiplyColumn<<<grid,block,0,stream>>>(outData,m_nInverseTable,modulii,mus);

		RelinPrepareCRTInput<<<grid,block,0,stream>>>(outData, m_rTableDev, modulii, mus);

	}

	cudaStreamSynchronize(stream);
}

__global__ void PreComputeRoot(uint32_t *table, CuBigInteger *modulus, CuBigInteger *root, CuBigInteger *mu, uint32_t n){

	uint32_t xArr[CuBigInteger::m_nSize];
	CuBigInteger x(CuBigInteger::Type::EMPTY);
	x.m_value = xArr;
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		x.m_value[i] = 0;
	}
	x.m_MSB = 1;
	x.m_value[CuBigInteger::m_nSize-1] = 1;

	for(uint32_t i=0; i<n; i++){
		for(uint32_t j = 0; j< CuBigInteger::m_nSize; j++){
			table[i*(CuBigInteger::m_nSize+1)+j] = x.m_value[j];
		}
		table[(i+1)*(CuBigInteger::m_nSize+1) -1] = x.m_MSB;
		x.ModBarrettMulInPlace(*root, *modulus, *mu);
	}
	x.m_value = nullptr;
}

__global__ void CRTPreComputeRoot(uint32_t *table, uint32_t *modulii, uint32_t *roots, uint32_t *mus, usint n){
	uint32_t crtIdx = blockIdx.x;
	uint32_t modulus = modulii[crtIdx];
	uint32_t root = roots[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];
	uint32_t x = 1;

	for(uint32_t i=0; i<n; i++){
		table[crtIdx*n + i] = x;
		x = ModBarretMult(x,root,modulus, mod2, mu,bits);
	}

}

__global__ void PrepareInput(uint32_t *inData, uint32_t *m_rTableDev, CuBigInteger *devModulus, CuBigInteger *devMu, uint32_t *outData, uint32_t *bitReverseLookup, uint32_t n){
	uint32_t bigintIdx = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t idx = bigintIdx*(CuBigInteger::m_nSize+1);

	uint32_t brevIdx = bitReverseLookup[bigintIdx];
	uint32_t outIdx = brevIdx*(CuBigInteger::m_nSize+1);

	uint32_t inpAtIArr[CuBigInteger::m_nSize];
	uint32_t rootAtIArr[CuBigInteger::m_nSize];
	CuBigInteger inpAtI(CuBigInteger::Type::EMPTY);
	CuBigInteger rootAtI(CuBigInteger::Type::EMPTY);

	inpAtI.m_value = inpAtIArr;
	inpAtI.m_MSB = inData[idx+CuBigInteger::m_nSize];
	rootAtI.m_value = rootAtIArr;
	rootAtI.m_MSB = m_rTableDev[idx+CuBigInteger::m_nSize];
	for(uint32_t i=0;i<CuBigInteger::m_nSize;i++){
		inpAtI.m_value[i] = inData[idx+i];
		rootAtI.m_value[i] = m_rTableDev[idx+i];
	}

	inpAtI.ModBarrettMulInPlace(rootAtI, *devModulus, *devMu);

	outData[outIdx+CuBigInteger::m_nSize] = inpAtI.m_MSB;
	for(uint32_t i=0;i<CuBigInteger::m_nSize;i++){
		outData[outIdx+i] = inpAtI.m_value[i];
	}

	inpAtI.m_value = nullptr;
	rootAtI.m_value = nullptr;

	//launch butterfly operations from thread 0 after synchronization
	__syncthreads();
	if(threadIdx.x==0){
		uint32_t log2n = log2((float)n);
		Butterfly3<<<1,n>>>(outData, log2n, m_rTableDev, devModulus, devMu);
	}
}

__global__ void PrepareCRTInput(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData, uint32_t *bitRevTable, uint32_t n){
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = blockIdx.x*n + threadIdx.x;
	uint32_t brevIdx = bitRevTable[threadIdx.x];
	uint32_t outIdx = blockIdx.x*n + brevIdx;
	uint32_t value = inData[idx];
	uint32_t root = table[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];

	uint32_t outValue = ModBarretMult(value, root, modulus, mod2, mu, bits);
	outData[outIdx] = outValue;

}

__global__ void PrepareCRTInputNew(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus){
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t value = inData[idx];
	uint32_t root = table[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];


	inData[idx] = ModBarretMult(value, root, modulus, mod2, mu, bits);
}

__global__ void PrepareCRTInputNewLarge(uint32_t *inData, uint32_t *table, uint32_t *modulii, uint32_t *mus){
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + threadIdx.x;
	uint32_t value = inData[idx];
	uint32_t root = table[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];

	inData[idx] = ModBarretMult(value, root, modulus, mod2, mu, bits);
}

__global__ void RelinPrepareCRTInput(uint32_t *data, uint32_t *table, uint32_t *modulii,uint32_t *mus){
	uint32_t crtIdx = blockIdx.y;
	uint32_t n = blockDim.x*gridDim.z;
	uint32_t block = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;
	uint32_t modulus = modulii[crtIdx];
	uint32_t bits = modulii[crtIdx + gridDim.y];
	uint32_t mod2 = modulii[crtIdx + 2*gridDim.y];
	uint32_t mu = mus[crtIdx];
	uint32_t root = table[crtIdx*n + blockIdx.z*blockDim.x + threadIdx.x];
	data[idx] = ModBarretMult(data[idx],root,modulus, mod2, mu,bits);
}

__global__ void RelinCRTButterfly(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *bitRevTable, uint32_t *outData){
	uint32_t n = blockDim.x;
	uint32_t crtIdx = blockIdx.y;
	uint32_t idx = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + threadIdx.x;

	uint32_t brevIdx = bitRevTable[threadIdx.x];

	uint32_t outIdx = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + brevIdx;

	//uint32_t root = table[crtIdx*blockDim.x + threadIdx.x];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.y+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.y+crtIdx];


	outData[outIdx] = inData[idx];

	__syncthreads();

	__shared__ uint32_t lCopy[1024];

	uint32_t omega,bflyMinus,bflyResult;

	lCopy[threadIdx.x] = bflyResult = outData[idx];
	__syncthreads(); //allow all threads to complete before computation

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	for(uint32_t stage=1; stage <= log2n; stage++){
		jump = (1<<stage);
		jumpHalf = (1<<(stage-1));

		checkBfly = threadIdx.x%jump;
		omegaIdx = (threadIdx.x%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega = table[crtIdx*n + omegaIdx];

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = threadIdx.x - jumpHalf;
			oddIdx = threadIdx.x;

			if(bflyResult>0){
				omega = ModBarretMult(omega,bflyResult,modulus, mod2, mu, bits);
				//load evenIdx in bflyResult
				bflyResult = lCopy[evenIdx];
				bflyResult = ModSub(bflyResult,omega, modulus);
			}
			else{
				bflyResult = lCopy[evenIdx];
			}

		}

		else{//this performs the butterfly + computation
			evenIdx = threadIdx.x;
			oddIdx = threadIdx.x + jumpHalf;
			//bflyMinus.m_MSB = lCopy[(oddIdx+1)*intSize-1];
			bflyMinus = lCopy[oddIdx];

			if(bflyMinus > 0){
				omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
				bflyResult = ModAdd(bflyResult, omega,modulus);
			}
		}

		__syncthreads();

		lCopy[threadIdx.x] = bflyResult;

		__syncthreads();

	}

	//finally write back to global memory
	outData[idx] = lCopy[threadIdx.x];
}

__global__ void RelinCRTButterflyLargeStage10(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus){
	uint32_t n = blockDim.x * gridDim.z;
	uint32_t polyStartIdx = (gridDim.y*gridDim.z*blockIdx.x + blockIdx.y*gridDim.z)*blockDim.x; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.z*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	//uint32_t root = table[idx];
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	__shared__ uint32_t lCopy[1024];

	uint32_t omega, bflyMinus, bflyResult;
	lCopy[threadIdx.x] = bflyResult = inData[idx];
	__syncthreads(); //allow all threads to complete before computation

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	for(uint32_t stage=1; stage <= 10; stage++){
		jump = (1 << stage);
		jumpHalf = (1 << (stage - 1));

		checkBfly = threadIdx.x%jump;
		omegaIdx = (threadIdx.x%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega = table[blockIdx.y*n + omegaIdx];

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = threadIdx.x - jumpHalf;
			oddIdx = threadIdx.x;

			if(bflyResult>0){
				omega = ModBarretMult(omega,bflyResult,modulus, mod2, mu, bits);
				//load evenIdx in bflyResult
				bflyResult = lCopy[evenIdx];
				bflyResult = ModSub(bflyResult,omega, modulus);
			}
			else{
				bflyResult = lCopy[evenIdx];
			}

		}

		else{//this performs the butterfly + computation
			evenIdx = threadIdx.x;
			oddIdx = threadIdx.x + jumpHalf;
			bflyMinus = lCopy[oddIdx];

			if(bflyMinus > 0){
				omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
				bflyResult = ModAdd(bflyResult, omega,modulus);
			}
		}

		__syncthreads();

		lCopy[threadIdx.x] = bflyResult;

		__syncthreads();

	}

	__syncthreads();
	//finally write back to global memory
	inData[idx] = lCopy[threadIdx.x];

}


__global__ void ScalarMultiply(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus){
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t value = data[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t nI = nInverseTable[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];

	data[idx] = ModBarretMult(value, nI, modulus, mod2, mu, bits);
}

__global__ void ScalarMultiplyLarge(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus){
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + threadIdx.x;
	uint32_t value = data[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t nI = nInverseTable[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];

	data[idx] = ModBarretMult(value, nI, modulus, mod2, mu, bits);
}

__global__ void ScalarMultiplyColumn(uint32_t *data, uint32_t *nInverseTable, uint32_t *modulii, uint32_t *mus){

	uint32_t block = blockIdx.z + blockIdx.y*gridDim.z + blockIdx.x*gridDim.y*gridDim.z;
	uint32_t idx = block*blockDim.x + threadIdx.x;
	uint32_t value = data[idx];
	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t nI = nInverseTable[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	data[idx] = ModBarretMult(value, nI, modulus, mod2, mu, bits);
}

__global__ void Butterfly(uint32_t *data, uint32_t jump, uint32_t n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu){
	uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=n){
		return;
	}
	uint32_t checkBfly = idx%jump;
	uint32_t jumpHalf = (jump>>1);
	uint32_t omegaIdx = (idx%jumpHalf)*(2*n/jump);
	uint32_t evenIdx, oddIdx;
	CuBigInteger omega(CuBigInteger::Type::EMPTY);
	CuBigInteger bfly(CuBigInteger::Type::EMPTY);
	uint32_t arr[(CuBigInteger::m_nSize+1)*2];
	omega.m_value = arr;
	bfly.m_value = arr + (CuBigInteger::m_nSize+1);
	omega.m_MSB = rTable[(omegaIdx+1)*(CuBigInteger::m_nSize+1)-1];
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		omega.m_value[i] = rTable[omegaIdx*(CuBigInteger::m_nSize+1)+i];
	}
	if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
		evenIdx = idx - jumpHalf;
		oddIdx = idx;
		bfly.m_MSB = data[(oddIdx+1)*(CuBigInteger::m_nSize+1)-1];

		if(bfly.m_MSB>0){
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[oddIdx*(CuBigInteger::m_nSize+1)+i];
			}
			omega.ModBarrettMulInPlace(bfly,*modulus,*mu);
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
			bfly.ModBarrettSubInPlace(omega,*modulus, *mu);
		}
		else{
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
		}

	}
	else{//this performs the butterfly + computation
		evenIdx = idx;
		oddIdx = idx + jumpHalf;
		bfly.m_MSB = data[(oddIdx+1)*(CuBigInteger::m_nSize+1)-1];
		if(bfly.m_MSB>0){
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[oddIdx*(CuBigInteger::m_nSize+1)+i];
			}
			omega.ModBarrettMulInPlace(bfly,*modulus,*mu);
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
			bfly.ModBarrettAddInPlace(omega,*modulus, *mu);
		}
		else{
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
		}

	}

	//write back the data
	data[(idx+1)*(CuBigInteger::m_nSize+1)-1] = bfly.m_MSB;
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		data[idx*(CuBigInteger::m_nSize+1)+i] = bfly.m_value[i];
	}
}

__global__ void Butterfly2(uint32_t *data, uint32_t stage, uint32_t log2n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu){

	if (stage > log2n)
		return; //base case

	uint32_t idx = threadIdx.x;

	uint32_t jump = (1<<stage);
	uint32_t jumpHalf = (1<<(stage-1));

	uint32_t checkBfly = idx%jump;
	uint32_t omegaIdx = (idx%jumpHalf)*(1<<(1+log2n-stage)); //
	uint32_t evenIdx, oddIdx;
	CuBigInteger omega(CuBigInteger::Type::EMPTY);
	CuBigInteger bfly(CuBigInteger::Type::EMPTY);

	uint32_t arr[(CuBigInteger::m_nSize+1)*2]; //reserve memory for omega and butterfly
	omega.m_value = arr;
	bfly.m_value = arr + (CuBigInteger::m_nSize+1);
	omega.m_MSB = rTable[(omegaIdx+1)*(CuBigInteger::m_nSize+1)-1];
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		omega.m_value[i] = rTable[omegaIdx*(CuBigInteger::m_nSize+1)+i];
	}
	if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
		evenIdx = idx - jumpHalf;
		oddIdx = idx;
		bfly.m_MSB = data[(oddIdx+1)*(CuBigInteger::m_nSize+1)-1];

		if(bfly.m_MSB>0){
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[oddIdx*(CuBigInteger::m_nSize+1)+i];
			}
			omega.ModBarrettMulInPlace(bfly,*modulus,*mu);
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
			bfly.ModBarrettSubInPlace(omega,*modulus, *mu);
		}
		else{
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
		}

	}
	else{//this performs the butterfly + computation
		evenIdx = idx;
		oddIdx = idx + jumpHalf;
		bfly.m_MSB = data[(oddIdx+1)*(CuBigInteger::m_nSize+1)-1];
		if(bfly.m_MSB>0){
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[oddIdx*(CuBigInteger::m_nSize+1)+i];
			}
			omega.ModBarrettMulInPlace(bfly,*modulus,*mu);
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
			bfly.ModBarrettAddInPlace(omega,*modulus, *mu);
		}
		else{
			bfly.m_MSB = data[(evenIdx+1)*(CuBigInteger::m_nSize+1)-1];
			for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
				bfly.m_value[i] = data[evenIdx*(CuBigInteger::m_nSize+1)+i];
			}
		}

	}

	//write back the data
	data[(idx+1)*(CuBigInteger::m_nSize+1)-1] = bfly.m_MSB;
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		data[idx*(CuBigInteger::m_nSize+1)+i] = bfly.m_value[i];
	}

	//release memory
	omega.m_value = nullptr;
	bfly.m_value = nullptr;

	__syncthreads();
	//launch the next Butterfly from thread 0
	if(idx==0){
		Butterfly2<<<1,(1<<log2n)>>>(data, stage+1, log2n, rTable, modulus, mu);
	}
}

__global__ void Butterfly3(uint32_t *data, uint32_t log2n, uint32_t *rTable, CuBigInteger *modulus, CuBigInteger *mu){

	uint32_t idx = threadIdx.x;
	const uint32_t intSize = (CuBigInteger::m_nSize+1);

	__shared__ uint32_t lCopy[intSize*1024];
	uint32_t mem[3*CuBigInteger::m_nSize];

	CuBigInteger omega(CuBigInteger::Type::EMPTY);
	CuBigInteger bflyMinus(CuBigInteger::Type::EMPTY);
	CuBigInteger bflyResult(CuBigInteger::Type::EMPTY);

	omega.m_value = mem;
	bflyMinus.m_value = omega.m_value + CuBigInteger::m_nSize;
	bflyResult.m_value = bflyMinus.m_value + CuBigInteger::m_nSize;


	//read from global memory into localMemory
	lCopy[(idx+1)*intSize-1] = bflyResult.m_MSB = data[(idx+1)*intSize-1];
	for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
		lCopy[idx*intSize + i] = bflyResult.m_value[i] = data[idx*intSize + i];
	}
	__syncthreads();

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	for(uint32_t stage=1; stage <= log2n; stage++){
		jump = (1<<stage);
		jumpHalf = (1<<(stage-1));

		checkBfly = idx%jump;
		omegaIdx = (idx%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega.m_MSB = rTable[(omegaIdx+1)*intSize-1];
		for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
			omega.m_value[i] = rTable[omegaIdx*intSize+i];
		}

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = idx - jumpHalf;
			oddIdx = idx;

			if(bflyResult.m_MSB>0){
				omega.ModBarrettMulInPlace(bflyResult,*modulus,*mu);
				bflyResult.m_MSB = lCopy[(evenIdx+1)*intSize-1];
				for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
					bflyResult.m_value[i] = lCopy[evenIdx*intSize+i];
				}
				bflyResult.ModBarrettSubInPlace(omega,*modulus, *mu);
			}
			else{
				bflyResult.m_MSB = lCopy[(evenIdx+1)*intSize-1];
				for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
					bflyResult.m_value[i] = lCopy[evenIdx*intSize+i];
				}
			}

		}
		else{//this performs the butterfly + computation
			evenIdx = idx;
			oddIdx = idx + jumpHalf;
			bflyMinus.m_MSB = lCopy[(oddIdx+1)*intSize-1];
			if(bflyMinus.m_MSB>0){
				for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
					bflyMinus.m_value[i] = lCopy[oddIdx*intSize+i];
				}
				omega.ModBarrettMulInPlace(bflyMinus,*modulus,*mu);
				bflyResult.ModBarrettAddInPlace(omega,*modulus, *mu);
			}

		}

		__syncthreads();

		lCopy[(idx+1)*intSize -1] = bflyResult.m_MSB;
		for(uint32_t i=0; i< CuBigInteger::m_nSize; i++){
			lCopy[idx*intSize+i] = bflyResult.m_value[i];
		}

		__syncthreads();
	}

	//write back memory to global memory;
	for(uint32_t i=0; i< intSize; i++){
		data[idx*intSize + i] = lCopy[idx*intSize + i];
	}

	//release memory
	omega.m_value = nullptr;
	bflyMinus.m_value = nullptr;
	bflyResult.m_value = nullptr;
}

__global__ void CRTButterflyNew(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *bitRevTable, uint32_t *outData){

	uint32_t n = blockDim.x;
	uint32_t crtIdx = blockIdx.x;
	uint32_t idx = crtIdx*n + threadIdx.x;

	uint32_t brevIdx = bitRevTable[threadIdx.x];
	uint32_t outIdx = crtIdx*n + brevIdx;

	//uint32_t root = table[idx];
	uint32_t modulus = modulii[crtIdx];
	uint32_t mu = mus[crtIdx];
	uint32_t bits = modulii[gridDim.x+crtIdx];
	uint32_t mod2 = modulii[2*gridDim.x+crtIdx];

	outData[outIdx] = inData[idx];

	__syncthreads();

	__shared__ uint32_t lCopy[1024];

	uint32_t omega,bflyMinus,bflyResult;
	lCopy[threadIdx.x] = bflyResult = outData[idx];
	__syncthreads(); //allow all threads to complete before computation

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	for(uint32_t stage=1; stage <= log2n; stage++){
		jump = (1<<stage);
		jumpHalf = (1<<(stage-1));

		checkBfly = threadIdx.x%jump;
		omegaIdx = (threadIdx.x%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega = table[blockIdx.x*n + omegaIdx];

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = threadIdx.x - jumpHalf;
			oddIdx = threadIdx.x;

			if(bflyResult>0){
				omega = ModBarretMult(omega,bflyResult,modulus, mod2, mu, bits);
				//load evenIdx in bflyResult
				bflyResult = lCopy[evenIdx];
				bflyResult = ModSub(bflyResult,omega, modulus);
			}
			else{
				bflyResult = lCopy[evenIdx];
			}

		}

		else{//this performs the butterfly + computation
			evenIdx = threadIdx.x;
			oddIdx = threadIdx.x + jumpHalf;
			//bflyMinus.m_MSB = lCopy[(oddIdx+1)*intSize-1];
			bflyMinus = lCopy[oddIdx];

			if(bflyMinus > 0){
				omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu, bits);
				bflyResult = ModAdd(bflyResult, omega,modulus);
			}
		}

		__syncthreads();

		lCopy[threadIdx.x] = bflyResult;

		__syncthreads();

	}

	//finally write back to global memory
	outData[idx] = lCopy[threadIdx.x];

}

__global__ void CRTButterflyNewLarge(uint32_t *inData, uint32_t log2n, uint32_t stage, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData){
	uint32_t n = blockDim.x * gridDim.y;
	uint32_t polyStartIdx = blockIdx.x*n; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.y*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	//uint32_t root = table[idx];
	uint32_t modulus = modulii[blockIdx.x];
	uint32_t mu = mus[blockIdx.x];
	uint32_t bits = modulii[gridDim.x+blockIdx.x];
	uint32_t mod2 = modulii[2*gridDim.x+blockIdx.x];

	uint32_t omega, bflyMinus, bflyResult;
	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	jump = (1<<stage);
	jumpHalf = (1<<(stage-1));

	checkBfly = localPolyIdx%jump;
	omegaIdx = (localPolyIdx%jumpHalf)*(1<<(1+log2n-stage));


	//load omega from global memory
	omega = table[polyStartIdx + omegaIdx];

	if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
		evenIdx = localPolyIdx - jumpHalf;
		oddIdx = localPolyIdx;

		bflyMinus = inData[polyStartIdx + oddIdx];
		bflyResult = inData[polyStartIdx + evenIdx];

		if(bflyMinus>0){
			omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu, bits);
			//load evenIdx in bflyResult
			bflyResult = ModSub(bflyResult,omega, modulus);
		}

	}

	else{//this performs the butterfly + computation
		evenIdx = localPolyIdx;
		oddIdx = localPolyIdx + jumpHalf;

		bflyMinus = inData[polyStartIdx + oddIdx];
		bflyResult = inData[polyStartIdx + evenIdx];

		if(bflyMinus > 0){
			omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
			bflyResult = ModAdd(bflyResult, omega,modulus);
		}
	}

	outData[idx] = bflyResult;

}

__global__ void RelinCRTButterflyLarge(uint32_t *inData, uint32_t log2n, uint32_t stage, uint32_t *table, uint32_t *modulii, uint32_t *mus, uint32_t *outData){
	uint32_t n = blockDim.x * gridDim.z;
	uint32_t polyStartIdx = (gridDim.y*gridDim.z*blockIdx.x + blockIdx.y*gridDim.z)*blockDim.x; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.z*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	uint32_t modulus = modulii[blockIdx.y];
	uint32_t mu = mus[blockIdx.y];
	uint32_t bits = modulii[gridDim.y+blockIdx.y];
	uint32_t mod2 = modulii[2*gridDim.y+blockIdx.y];

	uint32_t omega, bflyMinus, bflyResult;
	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	jump = (1<<stage);
	jumpHalf = (1<<(stage-1));

	checkBfly = localPolyIdx%jump;
	omegaIdx = (localPolyIdx%jumpHalf)*(1<<(1+log2n-stage));


	//load omega from global memory
	omega = table[blockIdx.y*n + omegaIdx];

	if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
		evenIdx = localPolyIdx - jumpHalf;
		oddIdx = localPolyIdx;

		bflyMinus = inData[polyStartIdx + oddIdx];
		bflyResult = inData[polyStartIdx + evenIdx];

		if(bflyMinus>0){
			omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu, bits);
			//load evenIdx in bflyResult
			bflyResult = ModSub(bflyResult,omega, modulus);
		}

	}

	else{//this performs the butterfly + computation
		evenIdx = localPolyIdx;
		oddIdx = localPolyIdx + jumpHalf;

		bflyMinus = inData[polyStartIdx + oddIdx];
		bflyResult = inData[polyStartIdx + evenIdx];

		if(bflyMinus > 0){
			omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
			bflyResult = ModAdd(bflyResult, omega,modulus);
		}
	}

	outData[idx] = bflyResult;
}


__global__ void BitReverse(uint32_t *inData, uint32_t *bitRevTable, uint32_t *outData){
	uint32_t n = blockDim.x * gridDim.y;
	uint32_t polyStartIdx = blockIdx.x*n; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.y*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	uint32_t brevIdx = bitRevTable[localPolyIdx];
	uint32_t outIdx = polyStartIdx + brevIdx;

	outData[outIdx] = inData[idx];
}

__global__ void RelinBitReverse(uint32_t *inData, uint32_t *bitRevTable, uint32_t *outData){
	uint32_t polyStartIdx = (gridDim.y*gridDim.z*blockIdx.x + blockIdx.y*gridDim.z)*blockDim.x; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.z*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	uint32_t brevIdx = bitRevTable[localPolyIdx];
	uint32_t outIdx = polyStartIdx + brevIdx;

	outData[outIdx] = inData[idx];
}



__global__ void CRTButterflyNewLargeStage10(uint32_t *inData, uint32_t log2n, uint32_t *table, uint32_t *modulii, uint32_t *mus){

	uint32_t n = blockDim.x * gridDim.y;
	uint32_t polyStartIdx = blockIdx.x*n; // starting idx of the poly in the crt polynomial
	uint32_t localPolyIdx = blockIdx.y*blockDim.x + threadIdx.x; // idx is from 0...n-1
	uint32_t idx = polyStartIdx + localPolyIdx; // index of the coefficient in crt polynomial

	//uint32_t root = table[idx];
	uint32_t modulus = modulii[blockIdx.x];
	uint32_t mu = mus[blockIdx.x];
	uint32_t bits = modulii[gridDim.x+blockIdx.x];
	uint32_t mod2 = modulii[2*gridDim.x+blockIdx.x];

	__shared__ uint32_t lCopy[1024];

	uint32_t omega, bflyMinus, bflyResult;
	lCopy[threadIdx.x] = bflyResult = inData[idx];
	__syncthreads(); //allow all threads to complete before computation

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	for(uint32_t stage=1; stage <= 10; stage++){
		jump = (1<<stage);
		jumpHalf = (1<<(stage-1));

		checkBfly = threadIdx.x%jump;
		omegaIdx = (threadIdx.x%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega = table[polyStartIdx + omegaIdx];

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = threadIdx.x - jumpHalf;
			oddIdx = threadIdx.x;

			if(bflyResult>0){
				omega = ModBarretMult(omega,bflyResult,modulus, mod2, mu, bits);
				//load evenIdx in bflyResult
				bflyResult = lCopy[evenIdx];
				bflyResult = ModSub(bflyResult,omega, modulus);
			}
			else{
				bflyResult = lCopy[evenIdx];
			}

		}

		else{//this performs the butterfly + computation
			evenIdx = threadIdx.x;
			oddIdx = threadIdx.x + jumpHalf;
			bflyMinus = lCopy[oddIdx];

			if(bflyMinus > 0){
				omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
				bflyResult = ModAdd(bflyResult, omega,modulus);
			}
		}

		__syncthreads();

		lCopy[threadIdx.x] = bflyResult;

		__syncthreads();

	}

	__syncthreads();
	//finally write back to global memory
	inData[idx] = lCopy[threadIdx.x];

}


__global__ void CRTButterfly(uint32_t *data, uint32_t log2n, uint32_t *rTable, uint32_t *modulii, uint32_t *mus){
	uint32_t n = blockDim.x;
	uint32_t idx = blockIdx.x*n + threadIdx.x;

	__shared__ uint32_t lCopy[1024];

	uint32_t omega,bflyMinus,bflyResult;
	lCopy[threadIdx.x] = bflyResult = data[idx];
	__syncthreads(); //allow all threads to complete before computation

	uint32_t evenIdx, oddIdx, jump, jumpHalf;
	uint32_t omegaIdx, checkBfly;

	uint32_t modulus = modulii[blockIdx.x];
	uint32_t mu = mus[blockIdx.x];
	uint32_t bits = modulii[gridDim.x+blockIdx.x];
	uint32_t mod2 = modulii[2*gridDim.x+blockIdx.x];

	for(uint32_t stage=1; stage <= log2n; stage++){
		jump = (1<<stage);
		jumpHalf = (1<<(stage-1));

		checkBfly = threadIdx.x%jump;
		omegaIdx = (threadIdx.x%jumpHalf)*(1<<(1+log2n-stage));


		//load omega from global memory
		omega = rTable[blockIdx.x*n + omegaIdx];

		if(checkBfly >= jumpHalf){ //this performs the butterfly - computation
			evenIdx = threadIdx.x - jumpHalf;
			oddIdx = threadIdx.x;

			if(bflyResult>0){
				omega = ModBarretMult(omega,bflyResult,modulus, mod2, mu, bits);
				//load evenIdx in bflyResult
				bflyResult = lCopy[evenIdx];
				bflyResult = ModSub(bflyResult,omega, modulus);
			}
			else{
				bflyResult = lCopy[evenIdx];
			}

		}

		else{//this performs the butterfly + computation
			evenIdx = threadIdx.x;
			oddIdx = threadIdx.x + jumpHalf;
			//bflyMinus.m_MSB = lCopy[(oddIdx+1)*intSize-1];
			bflyMinus = lCopy[oddIdx];

			if(bflyMinus > 0){
				omega = ModBarretMult(omega,bflyMinus,modulus, mod2, mu,bits);
				bflyResult = ModAdd(bflyResult, omega,modulus);
			}
		}

		__syncthreads();

		lCopy[threadIdx.x] = bflyResult;

		__syncthreads();

	}

	//finally write back to global memory
	data[blockIdx.x * n + threadIdx.x] = lCopy[threadIdx.x];
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
		printf("%u ", hostData[i * intSize + 2]);
	}
	printf("\n");

	delete []hostData;
}

void printCRTData(uint32_t *data, uint32_t dim, uint32_t crtSize){

	uint32_t *hostData = new uint32_t[dim*crtSize];

	cudaError_t err = cudaMemcpy(hostData, data, dim*crtSize*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "error in memory transfer\n";
	}

	for(uint32_t i=0; i<crtSize; i++){
		for(uint32_t j=0; j< dim; j++){
			std::cout << hostData[i*dim + j] << " ";
		}
		std::cout << '\n' ;
	}

	printf("\n");

	delete []hostData;
}


