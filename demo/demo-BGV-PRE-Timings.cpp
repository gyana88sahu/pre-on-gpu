
#include "../src/lattice/CuCRTParams.h"
#include "../src/lattice/CuCRTPoly.h"
#include "../src/pke/bgv-ingredient.h"
#include "../src/pke/cu-bgv.h"
#include "palisade.h"
#include <vector>

using namespace lbcrypto;
using namespace std;

void PRE_RUN(const std::shared_ptr<BGV_CryptoParameters> cp);
void NoiseGen(const std::shared_ptr<BGV_CryptoParameters> cp);

static shared_ptr<CuCRTParams> GetParamsFromFile(usint cyclo, usint bits, usint depth, string fileName);


int main(int argc, char **argv) {

	if(argc !=6){
		cout << "Wrong format of input, please feed proper inputs as follows \n ";
		cout << argv[0] << " cyclotomicNumber bits depth relinWindow fileName\n";
		cout << "exiting program\n";
		return -1;
	}

	usint m = atoi(argv[1]);
	usint bits = atoi(argv[2]);
	usint depth = atoi(argv[3]);
	usint r = atoi(argv[4]);
	string fileName(argv[5]);
	auto ep = GetParamsFromFile(m, bits, depth, fileName);


	uint32_t p = 5; //plaintext factor
	auto cp = std::make_shared<BGV_CryptoParameters>(ep,p,r); //cryto parameter
	//precomputation code is in Noise Gen
	NoiseGen(cp);
	cudaDeviceSynchronize();
	//main pre function is run after precompuation
	PRE_RUN(cp);
	cudaDeviceReset();
}

void PRE_RUN(const std::shared_ptr<BGV_CryptoParameters> cp){
	double start,stop;
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	auto ep = cp->GetElementParams();

	//KeyGen
	std::cout << "starting key generation for producer \n";
	start = currentDateTime();
	auto kp = CuBGV::KeyGen(cp);
	stop = currentDateTime();
	std::cout << "finished key generation\n";
	std::cout << "time taken for key generation is \t" << (stop-start) << " ms\n";

	//KeyGen
	std::cout << "starting key generation for consumer \n";
	start = currentDateTime();
	auto kpNew = CuBGV::KeyGen(cp);
	stop = currentDateTime();
	std::cout << "finished key generation\n";
	std::cout << "time taken for key generation is \t" << (stop-start) << " ms\n";

	CuCRTPoly mssg(ep, CuCRTPoly::COEFFICIENT, CuCRTPoly::BINARY, &stream);
	cudaStreamSynchronize(stream);
	cout << "message to be encrypted is\n" << mssg << '\n';

	std::cout << "starting encryption from producer public key \n";
	start = currentDateTime();
	auto cipher = CuBGV::Encrypt(kp.publicKey,mssg);
	stop = currentDateTime();
	std::cout << "finished encryption\n";
	std::cout << "time taken for encryption is \t" << (stop-start) << " ms\n";

	std::cout << "starting decryption from producer secret key \n";
	start = currentDateTime();
	auto decrypt = CuBGV::Decrypt(cipher, kp.secretKey);
	stop = currentDateTime();
	cout << decrypt << "\n";
	std::cout << "time taken for decryption is \t" << (stop-start) << " ms\n";

	//generate evaluation key
	std::cout << "starting ReKeyGen operation from producer secret key and consumer public key\n";
	start = currentDateTime();
	auto rk = CuBGV::ReKeyGen(kpNew.publicKey, kp.secretKey);
	stop = currentDateTime();
	std::cout << "finished ReKeyGen operation\n";
	std::cout << "time taken for ReKeyGen is \t" << (stop-start) << " ms\n";

	std::cout << "starting ReEncryption operation using the re-encryption key generated \n";
	start = currentDateTime();
	auto cipherNew = CuBGV::ReEncrypt(rk, cipher);
	stop = currentDateTime();
	std::cout << "finished ReEncryption operation\n";
	std::cout << "time taken for ReEncryption operation is \t" << (stop-start) << " ms\n";

	std::cout << "starting decryption from consumer secret key \n";
	start = currentDateTime();
	auto decryptNew = CuBGV::Decrypt(cipherNew, kpNew.secretKey);
	stop = currentDateTime();
	cout << decryptNew << "\n";
	std::cout << "time taken for decryption is \t" << (stop-start) << " ms\n";

	cudaStreamDestroy(stream);
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

shared_ptr<CuCRTParams> GetParamsFromFile(usint cyclo, usint bits, usint depth, string fileName){
	ifstream file(fileName);
	shared_ptr<CuCRTParams> params = nullptr;
	if (!file.is_open()) {
		cout << "file could not be opened\n";
		return params;
	}
	if (depth > 10) {
		cout << "depth upto 10 is only supported\n";
		return params;
	}
	if (bits > 30) {
		cout << "bits upto 30 is only supported\n";
		return params;
	}

	vector<uint32_t> modulii;
	vector<uint32_t> rootOfUnities;

	string line;
	while(getline(file, line)){
		istringstream iss(line);
		string first, second, third;
		iss >> first;

		if(first=="params"){
			iss >> second;
			iss >> third;
			if (second == to_string(cyclo) && third == to_string(bits)) {
				for (usint i = 0; i < depth; i++) {
					getline(file, line);
					istringstream iss1(line);
					iss1 >> first;
					iss1 >> second;
					uint32_t q = stoul(first, nullptr, 10);
					uint32_t root = stoul(second, nullptr, 10);
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
