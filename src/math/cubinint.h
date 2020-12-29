#ifndef LBCRYPTO_CRYPTO_CUBIGINT_DEFS_H
#define LBCRYPTO_CRYPTO_CUBIGINT_DEFS_H


#include <iostream>
#include <string>
#include <vector>
#include <string>
#include <typeinfo>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

using namespace std;

//class representing 64-bit integer, can be increased by increasing m_nSize=2
//Assume all arithmetic operations are performed on device
class CuBigInteger{

public:
	 enum Type
	{
	  EMPTY,
	  ALLOCATED
	 };

	const double LOG2_10 = 3.32192809;	//!< @brief A pre-computed constant of Log base 2 of 10.
	const uint32_t BARRETT_LEVELS = 8;		//!< @brief The number of levels (precomputed values) used in the Barrett reductions.

	__host__ __device__ CuBigInteger();

	__host__  __device__ CuBigInteger(const CuBigInteger &bigInteger);

	__host__  __device__ CuBigInteger(CuBigInteger &&bigInteger);

	__host__ CuBigInteger(const std::string& str);

	__host__ __device__ CuBigInteger(uint32_t num);

	__device__ CuBigInteger(CuBigInteger::Type t);

	__host__ static void InitializeDeviceVariables(CuBigInteger **ptr, std::string v);//allocates and initializes all non-pointer members

	__host__ static void DestroyDeviceVariables(CuBigInteger **ptr);

	__host__ static CuBigInteger GetHostRepresentation(CuBigInteger **ptr);

	__host__ static std::string ArrayToString(uint32_t *arr);

	__host__  __device__ const CuBigInteger&  operator=(const CuBigInteger &rhs);

	__host__  __device__ const CuBigInteger&  operator=(CuBigInteger &&rhs);

	__host__  __device__ const CuBigInteger& operator=(uint32_t val);

	__host__ void AssignVal(const std::string &v);

	__host__ __device__ ~CuBigInteger();

	__host__ __device__ void print() const;

	__host__ void transferDataToHost();

	__host__ __device__ static uint32_t GetMSB64(uint64_t x);

	__host__ __device__ void SetMSB();

	__host__ __device__ void SetMSB(uint32_t guessIdxChar);

	uint32_t *m_value;

	uint32_t m_MSB = 0;

	static const uint32_t m_uintBitLength = 32;

	static const uint32_t m_logUintBitLength = 5;

	static const uint32_t m_nSize = 6;

	static const uint32_t m_numDigitInPrintval = 50;

	static const uint32_t m_uintMax = 4294967295;

	static const CuBigInteger zero;


	__host__ __device__
	static uint32_t ceilIntByUInt(const uint32_t Number);

	__host__ __device__
	static uint32_t UintInBinaryToDecimal(uint32_t *a);

	__host__ __device__
	static void double_bitVal(uint32_t *a);

	__host__ __device__
	static void add_bitVal(uint32_t* a, uint32_t b);


	//arithmetic functions
	__device__ CuBigInteger Plus(const CuBigInteger& b) const;

	__device__ const CuBigInteger& PlusEq(const CuBigInteger& b);

	__device__
	inline friend CuBigInteger operator+(const CuBigInteger& a, const CuBigInteger& b) { return a.Plus(b); }

	__device__
	inline friend const CuBigInteger& operator+=(CuBigInteger& a, const CuBigInteger& b) { return a.PlusEq(b); }

	__device__ CuBigInteger Minus(const CuBigInteger& b) const;

	__device__ const CuBigInteger& MinusEq(const CuBigInteger& b);

	__device__
	inline friend CuBigInteger operator-(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Minus(b);
	}

	__device__
	inline friend const CuBigInteger& operator-=(CuBigInteger& a, const CuBigInteger& b) {
		return a.MinusEq(b);
	}


	__device__ CuBigInteger MulByUint(const uint32_t b) const;

	__device__ void MulByUintToInt(const uint32_t b, CuBigInteger* ans) const;

	__device__ CuBigInteger Times(const CuBigInteger& b) const;

	__device__ const CuBigInteger& TimesEq(const CuBigInteger& b);

	__device__
	inline friend CuBigInteger operator*(const CuBigInteger& a, const CuBigInteger& b) { return a.Times(b); }

	__device__
	inline friend const CuBigInteger& operator*=(CuBigInteger& a, const CuBigInteger& b) { return a.TimesEq(b); }


	__device__ CuBigInteger Mod(const CuBigInteger& modulus) const;

	__device__ void ModSelf(const CuBigInteger& modulus);

	__device__ void ModSubSelf(const CuBigInteger& b, const CuBigInteger& modulus);

	__device__ CuBigInteger ModBarrett(const CuBigInteger& modulus, const CuBigInteger& mu) const;

	__device__ void ModBarrettInPlace(const CuBigInteger& modulus, const CuBigInteger& mu);

	__device__ void ModBarrettMulInPlace(const CuBigInteger& b, const CuBigInteger& modulus, const CuBigInteger& mu);

	__device__ CuBigInteger ModBarrettMul(const CuBigInteger& b, const CuBigInteger& modulus,const CuBigInteger& mu) const;

	__device__ CuBigInteger ModBarrettAdd(const CuBigInteger& b, const CuBigInteger& modulus,const CuBigInteger &mu) const;

	__device__ void ModBarrettAddInPlace(const CuBigInteger& b, const CuBigInteger& modulus,const CuBigInteger &mu);

	__device__ void ModBarrettSubInPlace(const CuBigInteger& b, const CuBigInteger& modulus,const CuBigInteger &mu);

	__device__ CuBigInteger DividedBy(const CuBigInteger& b) const;

	__device__ CuBigInteger& DividedByEq(const CuBigInteger& b) ;

	__device__ int Compare(const CuBigInteger& a) const;

	__device__
	inline friend bool operator==(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) == 0;
	}

	__device__
	inline friend bool operator!=(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) != 0;
	}

	__device__
	inline friend bool operator>(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) > 0;
	}

	__device__
	inline friend bool operator>=(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) >= 0;
	}

	__device__
	inline friend bool operator<(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) < 0;
	}

	__device__
	inline friend bool operator<=(const CuBigInteger& a, const CuBigInteger& b) {
		return a.Compare(b) <= 0;
	}


	//shift operators
	__device__ CuBigInteger LShift(uint32_t shift) const;

	__device__ const CuBigInteger& LShiftEq(uint32_t shift);

	__device__
	inline friend CuBigInteger operator<<(const CuBigInteger& a, uint32_t shift) {
		return a.LShift(shift);
	}

	__device__
	inline friend const CuBigInteger& operator<<=(CuBigInteger& a, uint32_t shift) {
		return a.LShiftEq(shift);
	}

	__device__ CuBigInteger RShift(uint32_t shift) const;

	__device__ const CuBigInteger& RShiftEq(uint32_t shift);

	__device__
	inline friend CuBigInteger operator>>(const CuBigInteger& a, uint32_t shift) {
		return a.RShift(shift);
	}

	__device__
	inline friend const CuBigInteger& operator>>=(CuBigInteger& a, uint32_t shift) {
		return a.RShiftEq(shift);
	}

	__host__
	friend std::ostream& operator<<(std::ostream& os, const CuBigInteger& ptr_obj) {
		os << ptr_obj.ToString() ;
		return os;
	}

	__host__
	const std::string ToString() const;


	__host__ __device__ uint32_t GetBitAtIndex(uint32_t index) const;

	__host__ __device__ uint32_t GetDigitAtIndexForBase(uint32_t index, uint32_t baseBits) const;

};

#endif
