#include "CuCRTParams.h"

CuCRTParams::CuCRTParams(usint m, const std::vector<uint32_t> &modulii, const std::vector<uint32_t> &rootOfUnities){
	m_cycloNumber = m;
	m_ringDimension = m/2;
	uint32_t *hostModulii = new uint32_t[3*modulii.size()];
	uint32_t *hostRoots = new uint32_t[rootOfUnities.size()];
	m_hostModulii = modulii;
	m_hostRootOfUnities = rootOfUnities;
	for(usint i=0; i< modulii.size();i++){
		hostModulii[i] = modulii.at(i);
		uint32_t bits = std::ceil(std::log2((float)modulii.at(i)));
		hostModulii[i + modulii.size()] = bits;
		hostModulii[i + 2*modulii.size()] = modulii.at(i)*2;
		hostRoots[i] = rootOfUnities.at(i);
	}
	//allocating memory for modulii on device side
	cudaError_t err = cudaMalloc(&m_devModulii, 3*modulii.size()*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for devModulii in CRT Params\n";
		return;
	}
	//allocating memory for root of unity on device side
	err = cudaMalloc(&m_devRootOfUnities, rootOfUnities.size() * sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for devRootOfUnities in CRT Params\n";
		return;
	}
	//allocating memory for inverse root of unities on device side
	err = cudaMalloc(&m_devInverseRootOfUnities, rootOfUnities.size() * sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for devInverseRootOfUnities in CRT Params\n";
		return;
	}
	//allocating memory for mask on device side
	err = cudaMalloc(&m_devModuliiMask, modulii.size()*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for devModuliiMask in CRT Params\n";
		return;
	}
	//allocating memory for mus on device side
	err = cudaMalloc(&m_devMus, modulii.size()*sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for devMus in CRT Params\n";
		return;
	}

	//copying modulii memory from host to device
	err = cudaMemcpy(m_devModulii, hostModulii, 3*modulii.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for devModulii in CRT Params\n";
		return;
	}

	//copying roots memory from host to device
	err = cudaMemcpy(m_devRootOfUnities, hostRoots, rootOfUnities.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for devRootOfUnities in CRT Params\n";
		return;
	}

	uint32_t* inverseRoots = new uint32_t[rootOfUnities.size()];
	for(usint i=0; i< rootOfUnities.size();i++){
		NativeInteger num = rootOfUnities[i];
		num = num.ModInverse(hostModulii[i]);
		inverseRoots[i] = num.ConvertToInt();
	}
	//copying inverse roots memory from host to device
	err = cudaMemcpy(m_devInverseRootOfUnities, inverseRoots, rootOfUnities.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for devInverseRootOfUnities in CRT Params\n";
		return;
	}
	delete []inverseRoots;

	for(usint i=0; i< modulii.size();i++){
		uint32_t q = modulii[i];
		uint32_t bits = ceil(log2(q));
		uint32_t mask = 1 << (bits);
		mask -= 1;
		std::cout << "modulus is " << q << '\n';
		std::cout << "mask is " << mask << '\n';
		hostModulii[i] = mask;
	}
	//copying mask memory from host to device
	err = cudaMemcpy(m_devModuliiMask, hostModulii, modulii.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for devModuliiMask in CRT Params\n";
		return;
	}

	for(usint i=0; i< modulii.size();i++){
		uint32_t bits = hostModulii[modulii.size()+i];
		uint64_t powerOf2 = ((uint64_t)1 << (2*bits));
		uint32_t q = modulii[i];
		uint32_t mu = (uint32_t)(powerOf2 / q );
		hostModulii[i] = mu;
	}
	//copying mus memory from host to device
	err = cudaMemcpy(m_devMus , hostModulii, modulii.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for devMus in CRT Params\n";
		return;
	}

	delete []hostModulii;
	delete []hostRoots;

	curandCreateGenerator(&m_dug, CURAND_RNG_PSEUDO_MTGP32);
	curandCreateGenerator(&m_dgg, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(m_dug, time(0));
	curandSetPseudoRandomGeneratorSeed(m_dgg, time(0));

	bigModulus = BigInteger(1);
	for(usint i=0; i< modulii.size(); i++){
		bigModulus *= BigInteger(modulii[i]);
	}
	auto bigMu = ComputeMu<BigInteger>(bigModulus);

	CuBigInteger::InitializeDeviceVariables(&m_devBigModulus,bigModulus.ToString());
	CuBigInteger::InitializeDeviceVariables(&m_devBigMu,bigMu.ToString());


	usint bxiSize = (CuBigInteger::m_nSize+1)*m_hostModulii.size();
	//allocating memory for bxis on device side
	err = cudaMalloc(&m_bxis, bxiSize * sizeof(uint32_t));
	if (err != cudaSuccess) {
		std::cout << "error in cudaMalloc for bxis in CRT Params\n";
		return;
	}

	//populating m_bxis
	uint32_t *host_bxi = new uint32_t[bxiSize];
	for(usint i=0; i< modulii.size();i++){
		BigInteger modi(modulii[i]);
		BigInteger bi = bigModulus.DividedBy(modi);
		BigInteger xi = bi.ModInverse(modi);
		BigInteger bxi = bi*xi;
		//how are we going to break it down in chunks of 32 bits?
		CuBigInteger tempInt(bxi.ToString());
		host_bxi[(i+1)*(CuBigInteger::m_nSize+1)-1] = bxi.GetMSB();
		for (usint j = 0; j < CuBigInteger::m_nSize; j++) {
			host_bxi[i*(CuBigInteger::m_nSize+1) + j] = tempInt.m_value[j];
		}
	}

	err = cudaMemcpy(m_bxis, host_bxi, bxiSize*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "error in cudaMemcpy for bxis in CRT Params\n";
		return;
	}
	delete []host_bxi;
}

usint CuCRTParams::GetRingDimension() const{
	return m_ringDimension;
}

usint CuCRTParams::GetCyclotomicNumber() const{
	return m_cycloNumber;
}
