#include "rgsw-ingredient.h"

CuCRT_RGSW_CryptoParameters::CuCRT_RGSW_CryptoParameters(){
	m_params = nullptr;
	m_ptxtModulus = m_relinWindow = 0;
}

CuCRT_RGSW_CryptoParameters::CuCRT_RGSW_CryptoParameters(const shared_ptr<CuCRTParams> params, uint32_t p, uint32_t r){
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

shared_ptr<CuCRTParams> CuCRT_RGSW_CryptoParameters::GetElementParams(){
	return m_params;
}

uint32_t CuCRT_RGSW_CryptoParameters::GetPlaintextModulus(){
	return m_ptxtModulus;
}

uint32_t CuCRT_RGSW_CryptoParameters::GetRelinWindow(){
	return m_relinWindow;
}

uint32_t CuCRT_RGSW_CryptoParameters::GetDigits(){
	usint nBits = m_params->bigModulus.GetMSB();
	usint nDigits = 1;
	if (m_relinWindow > 0) {
		nDigits = nBits / m_relinWindow;
		if (nBits % m_relinWindow > 0){
			nDigits++;
		}
	}

	return nDigits;
}

uint32_t CuCRT_RGSW_CryptoParameters::GetRows(){
	uint32_t rows = GetDigits() << 1;
	return rows;
}

CuCRT_RGSW_Key::CuCRT_RGSW_Key(){
	m_cryptoParams = nullptr;
}

CuCRT_RGSW_Key::CuCRT_RGSW_Key(const shared_ptr<CuCRT_RGSW_CryptoParameters> params){
	m_cryptoParams = params;
}

shared_ptr<CuCRT_RGSW_CryptoParameters> CuCRT_RGSW_Key::GetCryptoParameters() const{
	return m_cryptoParams;
}

CuCRT_RGSW_SecretKey::CuCRT_RGSW_SecretKey(const shared_ptr<CuCRT_RGSW_CryptoParameters> params):
		CuCRT_RGSW_Key(params)	{}

const CuCRTPoly& CuCRT_RGSW_SecretKey::GetSecretKeyElement(){
	return m_sk;
}

void CuCRT_RGSW_SecretKey::SetSecretKeyElement(const CuCRTPoly &poly){
	m_sk = poly;
}

void CuCRT_RGSW_SecretKey::SetSecretKeyElement(CuCRTPoly &&poly){
	m_sk = std::move(poly);
}

CuCRT_RGSW_PublicKey::CuCRT_RGSW_PublicKey(const shared_ptr<CuCRT_RGSW_CryptoParameters> params) :
	CuCRT_RGSW_Key(params) {}

void CuCRT_RGSW_PublicKey::SetPublicKeyElementA(const CuCRTPoly &poly){
	 m_A = poly;
}
void CuCRT_RGSW_PublicKey::SetPublicKeyElementA(CuCRTPoly &&poly){
	m_A = std::move(poly);
}

const CuCRTPoly& CuCRT_RGSW_PublicKey::GetPublicKeyElementA(){
	return m_A;
}

void CuCRT_RGSW_PublicKey::SetPublicKeyElementB(const CuCRTPoly &poly){
	m_B = poly;
}
void CuCRT_RGSW_PublicKey::SetPublicKeyElementB(CuCRTPoly &&poly){
	m_B = std::move(poly);
}

const CuCRTPoly& CuCRT_RGSW_PublicKey::GetPublicKeyElementB(){
	return m_B;
}

CuCRT_RGSW_KeyPair::CuCRT_RGSW_KeyPair(const shared_ptr<CuCRT_RGSW_CryptoParameters> params){
	secretKey = make_shared<CuCRT_RGSW_SecretKey>(params);
	publicKey = make_shared<CuCRT_RGSW_PublicKey>(params);
}

CuCRT_RGSW_Ciphertext::CuCRT_RGSW_Ciphertext(const shared_ptr<CuCRT_RGSW_CryptoParameters> params): CuCRT_RGSW_Key(params){
	auto ep = params->GetElementParams();
	uint32_t rows = params->GetRows();
	cudaStream_t aStream, bStream;
	cudaStreamCreate(&aStream);
	cudaStreamCreate(&bStream);
	m_A = std::move(CuCRTColumnPoly(ep, CuCRTColumnPoly::Format::EVALUATION, rows, aStream));
	m_B = std::move(CuCRTColumnPoly(ep, CuCRTColumnPoly::Format::EVALUATION, rows, bStream));
	cudaStreamSynchronize(aStream);
	cudaStreamSynchronize(bStream);
	cudaStreamDestroy(aStream);
	cudaStreamDestroy(bStream);
}

CuCRT_RGSW_Ciphertext::CuCRT_RGSW_Ciphertext(const CuCRT_RGSW_Ciphertext &cipher): CuCRT_RGSW_Key(cipher.GetCryptoParameters()){
	m_A = cipher.GetElementA();
	m_B = cipher.GetElementB();
}

CuCRT_RGSW_Ciphertext::CuCRT_RGSW_Ciphertext(CuCRT_RGSW_Ciphertext &&cipher): CuCRT_RGSW_Key(cipher.GetCryptoParameters()){
	m_A = std::move(cipher.m_A);
	m_B = std::move(cipher.m_B);
}

void CuCRT_RGSW_Ciphertext::SetElementA(const CuCRTColumnPoly &colmnA){
	m_A = colmnA;
}

void CuCRT_RGSW_Ciphertext::SetElementA(const CuCRTColumnPoly &&colmnA){
	m_A = std::move(colmnA);
}

const CuCRTColumnPoly& CuCRT_RGSW_Ciphertext::GetElementA() const{
	return m_A;
}

void CuCRT_RGSW_Ciphertext::SetElementB(const CuCRTColumnPoly &colmnB){
	m_B = colmnB;
}

void CuCRT_RGSW_Ciphertext::SetElementB(const CuCRTColumnPoly &&colmnB){
	m_B = std::move(colmnB);
}

const CuCRTColumnPoly& CuCRT_RGSW_Ciphertext::GetElementB() const{
	return m_B;
}


CuCRT_RGSW_KeySwitchMatrix::CuCRT_RGSW_KeySwitchMatrix(const shared_ptr<CuCRT_RGSW_CryptoParameters> params): CuCRT_RGSW_Key(params){
	m_eval0 = make_shared<CuCRT_RGSW_Ciphertext>(params);
	m_eval1 = make_shared<CuCRT_RGSW_Ciphertext>(params);
}
