#ifndef LBCRYPTO_CU_CRT_POLY_DEFS_H
#define LBCRYPTO_CU_CRT_POLY_DEFS_H

#include "CuCRTParams.h"
#include "../transfrm/transform.h"
#include <curand.h>
#include <cuda.h>

using namespace lbcrypto;
using namespace std;


class CuCRTPoly{

public:
	enum Format {
		COEFFICIENT, EVALUATION
	};

	enum NoiseType {
		GAUSSIAN, UNIFORM, TERNARY, BINARY
	};

	CuCRTPoly();

	CuCRTPoly(const CuCRTPoly &poly);

	CuCRTPoly(CuCRTPoly &&poly);

	//assigns memory on device with all elements 0
	//and sets internal paramters
	CuCRTPoly(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format=COEFFICIENT);

	//assigns memory on device with all elements 0
	//and sets internal paramters
	CuCRTPoly(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format, CuCRTPoly::NoiseType noise, cudaStream_t *stream);

	//copy assignment operator
	CuCRTPoly& operator=(const CuCRTPoly &poly);

	//move assignment operator
	CuCRTPoly& operator=(CuCRTPoly &&poly);

	const CuCRTPoly& operator=(std::initializer_list<int> rhs);

	~CuCRTPoly();

	std::vector<NativePoly> ConvertToNativePoly() const;

	static std::vector<uint32_t> ConvertRelinPolyToVector(uint32_t *data, shared_ptr<CuCRTParams> ep, usint r);

	void SwitchFormat(cudaStream_t *stream=nullptr);

	void GenerateUniformDistribution(cudaStream_t *stream=nullptr);

	void GenerateGaussianDistribution(cudaStream_t *stream=nullptr);

	void GenerateTernaryDistribution(cudaStream_t *stream=nullptr);

	void GenerateBinaryDistribution(cudaStream_t *stream=nullptr);

	std::vector<uint32_t> SignedMod(uint32_t mod);

	void printDataFromDevice();

	uint32_t*  PowersOf2(uint32_t r,uint32_t* powersOf2, cudaStream_t *stream) const;

	uint32_t* BitDecompose(uint32_t r) const;

	CuCRTPoly Times(const CuCRTPoly &poly) const;

	CuCRTPoly Add(const CuCRTPoly &poly) const;

	CuCRTPoly Subtract(const CuCRTPoly &poly) const;

	CuCRTPoly Times(uint32_t num) const;

	const CuCRTPoly& TimesEq(uint32_t num);

	static uint32_t* GenerateRelinNoise(const shared_ptr<CuCRTParams> ep, CuCRTPoly::Format format, CuCRTPoly::NoiseType noise, usint polySize, cudaStream_t *stream);

	//m_data represents the crt polynomial interleaved by polynomial length
	uint32_t *m_data;
	uint32_t m_crtLength;
	std::shared_ptr<CuCRTParams> m_params = nullptr;
	CuCRTPoly::Format m_format;
	float *m_discrteGaussianDevData = nullptr;

	friend CuCRTPoly operator*(const CuCRTPoly &poly, uint32_t num){
		return std::move(poly.Times(num));
	}

	friend CuCRTPoly operator*(uint32_t num, const CuCRTPoly &poly ){
		return std::move(poly.Times(num));
	}

	friend CuCRTPoly operator*(const CuCRTPoly &poly1, const CuCRTPoly &poly2 ){
		return std::move(poly1.Times(poly2));
	}

	friend CuCRTPoly operator+(const CuCRTPoly &poly1, const CuCRTPoly &poly2 ){
		return std::move(poly1.Add(poly2));
	}

	friend CuCRTPoly operator-(const CuCRTPoly &poly1, const CuCRTPoly &poly2 ){
		return std::move(poly1.Subtract(poly2));
	}

	friend ostream& operator<<(ostream& os, const CuCRTPoly& poly){
		usint dim = poly.m_params->GetRingDimension();
		uint32_t *host = new uint32_t[poly.m_crtLength*dim];
		cudaError_t err = cudaMemcpy(host, poly.m_data, poly.m_crtLength*dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if(err!= cudaSuccess){
			throw std::runtime_error("error is transferring memory from device to host in printData function\n");
		}
		for(usint i=0; i< poly.m_crtLength; i++){
			for(usint j=0;j< dim; j++){
				os << host[i*dim +j] << " ";
			}
			os << '\n' << "<------\n";
		}
		os << '\n';
		delete []host;
		return os;
	}

};



#endif
