#ifndef LBCRYPTO_CUPOLY_DEFS_H
#define LBCRYPTO_CUPOLY_DEFS_H


#include "CuPolyParams.h"
#include "../../src/math/cubinint.h"
#include "../transfrm/transform.h"
#include <curand.h>
#include <cuda.h>

using namespace lbcrypto;

class CuPoly{

public:
    enum Format
    {
        COEFFICIENT,
        EVALUATION
    };

    enum NoiseType
    {
        GAUSSIAN,
        UNIFORM,
        TERNARY
    };
    
    CuPoly(){};
    
    //assigns memory on device with all elements 0 
    //and sets internal paramters
    CuPoly(const shared_ptr<CuPolyParams> ep, CuPoly::Format format);

    //assigns memory on device with all elements 0 
    //and sets internal paramters
    CuPoly(const shared_ptr<CuPolyParams> ep, CuPoly::Format format, CuPoly::NoiseType noise);

    void SwitchFormat();

    void GenerateUniformDistribution();

    void GenerateGaussianDistribution();

private:
    uint32_t *m_data;
    std::shared_ptr<CuPolyParams> m_params = nullptr;
    CuPoly::Format m_format;
    curandGenerator_t m_dug;
	curandGenerator_t m_dgg;
	float *m_discrteGaussianDevData = nullptr;
};

#endif
