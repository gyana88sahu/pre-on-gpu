
#include "bgv-ingredient.h"

BGV_CryptoParameters::BGV_CryptoParameters(){
	m_params = nullptr;
	m_ptxtModulus = m_relinWindow = 0;
}

BGV_CryptoParameters::BGV_CryptoParameters(const shared_ptr<CuCRTParams> params, uint32_t p, uint32_t r){
	m_params = params;
	m_ptxtModulus = p;
	m_relinWindow = r;

	usint nBits = params->bigModulus.GetMSB();
	usint nDigits = 1;
	if (r > 0) {
		nDigits = nBits / r;
		if (nBits % r > 0){
			nDigits++;
		}
	}
	uint32_t crtPowersLength = params->m_hostModulii.size()*nDigits;
	uint32_t relinPolyLength = crtPowersLength*params->GetRingDimension();
	cudaError_t err = cudaMalloc(&m_powersOf2, relinPolyLength*sizeof(uint32_t));
	if(err!=cudaSuccess){
		throw std::runtime_error("error in allocating memory for power of 2 polynomials\n");
	}

	uint32_t *hostPowersof2 = new uint32_t[crtPowersLength];
	for (usint i = 0; i < m_params->m_hostModulii.size(); i++) {
		BigInteger powerOf2(1);
		for (usint j = 0; j < nDigits; j++) {
			powerOf2 = powerOf2.Mod(BigInteger(m_params->m_hostModulii[i]));
			hostPowersof2[j*m_params->m_hostModulii.size() + i] = powerOf2.ConvertToInt();
			powerOf2 <<= r;
		}
	}
	err = cudaMemcpy(m_powersOf2, hostPowersof2, crtPowersLength*sizeof(uint32_t), cudaMemcpyHostToDevice);
	for(int i=0; i<crtPowersLength; i++){
		cout << hostPowersof2[i] << "  ";
	}
	cout << '\n';
	delete []hostPowersof2;
}

shared_ptr<CuCRTParams> BGV_CryptoParameters::GetElementParams(){
	return m_params;
}

uint32_t BGV_CryptoParameters::GetPlaintextModulus(){
	return m_ptxtModulus;
}

uint32_t BGV_CryptoParameters::GetRelinWindow(){
	return m_relinWindow;
}

CuCRT_BGV_Key::CuCRT_BGV_Key(){
	m_cryptoParams = nullptr;
}

CuCRT_BGV_Key::CuCRT_BGV_Key(const shared_ptr<BGV_CryptoParameters> params){
	m_cryptoParams = params;
}

shared_ptr<BGV_CryptoParameters> CuCRT_BGV_Key::GetCryptoParameters() const{
	return m_cryptoParams;
}

CuCRT_BGV_SecretKey::CuCRT_BGV_SecretKey(const shared_ptr<BGV_CryptoParameters> params):
		CuCRT_BGV_Key(params)	{}

const CuCRTPoly& CuCRT_BGV_SecretKey::GetSecretKeyElement(){
	return m_sk;
}

void CuCRT_BGV_SecretKey::SetSecretKeyElement(const CuCRTPoly &poly){
	m_sk = poly;
}

void CuCRT_BGV_SecretKey::SetSecretKeyElement(CuCRTPoly &&poly){
	m_sk = std::move(poly);
}

CuCRT_BGV_PublicKey::CuCRT_BGV_PublicKey(const shared_ptr<BGV_CryptoParameters> params) :
	CuCRT_BGV_Key(params) {}

void CuCRT_BGV_PublicKey::SetPublicKeyElementA(const CuCRTPoly &poly){
	 m_A = poly;
}
void CuCRT_BGV_PublicKey::SetPublicKeyElementA(CuCRTPoly &&poly){
	m_A = std::move(poly);
}

const CuCRTPoly& CuCRT_BGV_PublicKey::GetPublicKeyElementA(){
	return m_A;
}

void CuCRT_BGV_PublicKey::SetPublicKeyElementB(const CuCRTPoly &poly){
	m_B = poly;
}
void CuCRT_BGV_PublicKey::SetPublicKeyElementB(CuCRTPoly &&poly){
	m_B = std::move(poly);
}

const CuCRTPoly& CuCRT_BGV_PublicKey::GetPublicKeyElementB(){
	return m_B;
}

CuCRT_BGV_KeyPair::CuCRT_BGV_KeyPair(const shared_ptr<BGV_CryptoParameters> params){
	secretKey = make_shared<CuCRT_BGV_SecretKey>(params);
	publicKey = make_shared<CuCRT_BGV_PublicKey>(params);
}


CuCRT_BGV_Ciphertext::CuCRT_BGV_Ciphertext(const shared_ptr<BGV_CryptoParameters> params): CuCRT_BGV_Key(params){
	auto ep = params->GetElementParams();
	m_A = std::move(CuCRTPoly(ep, CuCRTPoly::Format::EVALUATION));
	m_B = std::move(CuCRTPoly(ep, CuCRTPoly::Format::EVALUATION));
}

CuCRT_BGV_Ciphertext::CuCRT_BGV_Ciphertext(const CuCRT_BGV_Ciphertext &cipher): CuCRT_BGV_Key(cipher.GetCryptoParameters()){
	m_A = cipher.GetElementA();
	m_B = cipher.GetElementB();
}

CuCRT_BGV_Ciphertext::CuCRT_BGV_Ciphertext(CuCRT_BGV_Ciphertext &&cipher): CuCRT_BGV_Key(cipher.GetCryptoParameters()){
	m_A = std::move(cipher.m_A);
	m_B = std::move(cipher.m_B);
}

void CuCRT_BGV_Ciphertext::SetElementA(const CuCRTPoly &poly){
	m_A = poly;
}

void CuCRT_BGV_Ciphertext::SetElementA(CuCRTPoly &&poly){
	m_A = std::move(poly);
}

const CuCRTPoly& CuCRT_BGV_Ciphertext::GetElementA() const {
	return m_A;
}

void CuCRT_BGV_Ciphertext::SetElementB(const CuCRTPoly &poly){
	m_B = poly;
}
void CuCRT_BGV_Ciphertext::SetElementB(CuCRTPoly &&poly){
	m_B = std::move(poly);
}

const CuCRTPoly& CuCRT_BGV_Ciphertext::GetElementB() const {
	return m_B;
}
