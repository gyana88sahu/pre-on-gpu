#include "palisade.h"
#include "scheme/rlwe.h"
#include "../src/lattice/CuPolyParams.h"
#include "../src/lattice/CuPoly.h"
#include "../src/lattice/CuCRTColumnPoly.h"
#include "../src/pke/rgsw-ingredient.h"
#include "../src/pke/bgv-ingredient.h"
#include "../src/pke/cu-rgsw.h"


using namespace lbcrypto;
using namespace std;

static void NoiseGen(const std::shared_ptr<BGV_CryptoParameters> cp);

static void KeyGenTest(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);

static void EncryptionTest(const shared_ptr<CuCRT_RGSW_CryptoParameters> params);

static shared_ptr<CuCRTParams> GetParamsFromFile(usint cyclo, usint depth, string fileName);


int main(int argc, char **argv) {

	if(argc !=4){
		cout << "Wrong format of input, please feed proper inputs as follows\n ";
		cout << argv[0] << "  cyclotomicNumber depth fileName\n";
		cout << "exiting program\n";
		return -1;
	}

	usint m = atoi(argv[1]);
	usint depth = atoi(argv[2]);
	string fileName(argv[3]);
	auto ep = GetParamsFromFile(m, depth, fileName);

	uint32_t p = 5; //plaintext factor
	uint32_t r = 1; //relin factor
	auto cp = std::make_shared<CuCRT_RGSW_CryptoParameters>(ep,p,r); //Ring-GSW crypto parameter
	auto cpBV = std::make_shared<BGV_CryptoParameters>(ep,p,r); //BV crypto parameter
	
	//precomputation code is in Noise Gen
	NoiseGen(cpBV);
	cudaDeviceSynchronize();
	KeyGenTest(cp);
	EncryptionTest(cp);
    cudaDeviceReset();
    return 0;
}

shared_ptr<CuCRTParams> GetParamsFromFile(usint cyclo, usint depth, string fileName){
	ifstream file(fileName);
	shared_ptr<CuCRTParams> params = nullptr;
	if(!file.is_open()){
		cout << "file could not be opened\n";
		return params;
	}
	if(depth>10){
		cout << "depth upto 10 is only supported\n";
		return params;
	}

	vector<uint32_t> modulii;
	vector<uint32_t> rootOfUnities;

	string line;
	while(getline(file, line)){
		istringstream iss(line);
		string first, second;
		iss >> first;

		if(first=="params"){
			iss >> second;
			if(second==to_string(cyclo)){
				for(usint i=0; i<depth; i++){
					getline(file, line);
					uint32_t q = stoul(line,nullptr, 10);
					getline(file, line);
					uint32_t root = stoul(line,nullptr, 10);
					modulii.push_back(q);
					rootOfUnities.push_back(root);
				}
				break;
			}
		}
	}

	params = std::make_shared<CuCRTParams>(cyclo, modulii, rootOfUnities);

	file.close();
	return params;
}

void KeyGenTest(const shared_ptr<CuCRT_RGSW_CryptoParameters> params){
	auto kp = CuRGSW::KeyGen(params);
}

void NoiseGen(const std::shared_ptr<BGV_CryptoParameters> cp){
	auto params = cp->GetElementParams();
	cudaStream_t aStream;
	cudaStreamCreate(&aStream);
	CuCRTPoly a(params, CuCRTPoly::Format::EVALUATION, CuCRTPoly::NoiseType::UNIFORM, &aStream);
	a.SwitchFormat(&aStream);
	a.SwitchFormat(&aStream);
	cudaStreamSynchronize(aStream);
	cudaStreamDestroy(aStream);
}

void EncryptionTest(const shared_ptr<CuCRT_RGSW_CryptoParameters> params){
	auto kp = CuRGSW::KeyGen(params);
	auto ep = params->GetElementParams();
	cudaStream_t mssgStream;
	cudaStreamCreate(&mssgStream);
	CuCRTPoly mssg(ep, CuCRTPoly::Format::COEFFICIENT, CuCRTPoly::NoiseType::BINARY, &mssgStream);
	cudaStreamSynchronize(mssgStream);
	cudaStreamDestroy(mssgStream);
	cout << mssg << endl;

	auto cipher = CuRGSW::Encrypt(kp.publicKey, mssg);

	auto ptxt = CuRGSW::Decrypt(cipher, kp.secretKey);

	cout << ptxt << endl;
}
