#ifndef LBCRYPTO_CU_CRT_COLUMN_POLY_DEFS_H
#define LBCRYPTO_CU_CRT_COLUMN_POLY_DEFS_H

#include "CuCRTParams.h"
#include "CuCRTPoly.h"
#include "../transfrm/transform.h"
#include <curand.h>
#include <cuda.h>

using namespace lbcrypto;
using namespace std;


class CuCRTColumnPoly{

public:
	enum Format {
		COEFFICIENT, EVALUATION
	};

	enum NoiseType {
		GAUSSIAN, UNIFORM, TERNARY, BINARY
	};

	CuCRTColumnPoly();

	CuCRTColumnPoly(const CuCRTColumnPoly &colmn);

	CuCRTColumnPoly(CuCRTColumnPoly &&colmn);

	//assigns memory on device with all elements 0
	//and sets internal paramters
	CuCRTColumnPoly(const shared_ptr<CuCRTParams> ep, CuCRTColumnPoly::Format format, usint rows, cudaStream_t &stream);

	//assigns memory on device with all elements 0
	//and sets internal paramters
	CuCRTColumnPoly(const shared_ptr<CuCRTParams> ep, CuCRTColumnPoly::Format format, CuCRTColumnPoly::NoiseType noise, usint rows, cudaStream_t &stream);

	//copy assignment operator
	CuCRTColumnPoly& operator=(const CuCRTColumnPoly &colmn);

	//move assignment operator
	CuCRTColumnPoly& operator=(CuCRTColumnPoly &&colmn);

	const CuCRTColumnPoly& operator=(std::initializer_list<int> rhs);

	~CuCRTColumnPoly();

	void SwitchFormat(cudaStream_t &stream);

	void GenerateUniformDistribution(cudaStream_t &stream);

	void GenerateGaussianDistribution(cudaStream_t &stream);

	void GenerateTernaryDistribution(cudaStream_t &stream);

	void GenerateBinaryDistribution(cudaStream_t &stream);

	CuCRTColumnPoly Times(const CuCRTPoly &poly) const;

	CuCRTColumnPoly Times(const CuCRTColumnPoly &colmn) const;

	CuCRTColumnPoly Times(uint32_t num) const;

	CuCRTColumnPoly Add(const CuCRTColumnPoly &colmn) const;

	void AddEq(const CuCRTColumnPoly &colmn);

	CuCRTColumnPoly Subtract(const CuCRTColumnPoly &colmn) const;

	void SubtractEq(const CuCRTColumnPoly &colmn);

	CuCRTPoly Reduce(cudaStream_t &stream) ;

	//m_data represents the crt polynomial interleaved by polynomial length
	uint32_t *m_data = nullptr;
	uint32_t m_crtLength;
	uint32_t m_rows;
	std::shared_ptr<CuCRTParams> m_params = nullptr;
	CuCRTColumnPoly::Format m_format;
	float *m_discrteGaussianDevData = nullptr;

	CuCRTPoly operator[](uint32_t row);

	const CuCRTPoly operator[](uint32_t row) const;

	friend CuCRTColumnPoly operator*(const CuCRTColumnPoly &colmn, uint32_t num){
		return std::move(colmn.Times(num));
	}

	friend CuCRTColumnPoly operator*(uint32_t num, const CuCRTColumnPoly &colmn ){
		return std::move(colmn.Times(num));
	}

	friend CuCRTColumnPoly operator*(const CuCRTColumnPoly &colmn, const CuCRTPoly &poly ){
		return std::move(colmn.Times(poly));
	}

	friend CuCRTColumnPoly operator*(const CuCRTPoly &poly, const CuCRTColumnPoly &colmn ){
		return std::move(colmn.Times(poly));
	}

	friend CuCRTColumnPoly operator*(const CuCRTColumnPoly &colmn1, const CuCRTColumnPoly &colmn2 ){
		return std::move(colmn1.Times(colmn2));
	}

	friend CuCRTColumnPoly operator+(const CuCRTColumnPoly &colmn1, const CuCRTColumnPoly &colmn2 ){
		return std::move(colmn1.Add(colmn2));
	}

	friend CuCRTColumnPoly& operator+=(CuCRTColumnPoly &colmn1, const CuCRTColumnPoly &colmn2 ){
		colmn1.AddEq(colmn2);
		return colmn1;
	}

	friend CuCRTColumnPoly operator-(const CuCRTColumnPoly &colmn1, const CuCRTColumnPoly &colmn2 ){
		return std::move(colmn1.Subtract(colmn2));
	}

	friend CuCRTColumnPoly& operator-=(CuCRTColumnPoly &colmn1, const CuCRTColumnPoly &colmn2 ){
		colmn1.SubtractEq(colmn2);
		return colmn1;
	}

	static CuCRTColumnPoly GetBitDecomposedColumnPoly(const CuCRTPoly &x, const CuCRTPoly &y, uint32_t rows, uint32_t relinWindow);

};



#endif
