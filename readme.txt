This project serves as the implementation of NTT, CRT and Proxy-Re-Encryption algorithms on GPUs
Purpose of this project:
1. Implementation of GPU arbitrary precision integers
2. Implementation of Number Theoretic Transform on GPUs
3. Implementation of Chinese Remainder Transform of Polynomials on GPUs
4. Implementation of BitDecomposition Algorithm on Ring-LWE Polynomials.
5. Implementation of BV-PRE and Ring-GSW-PRE on GPUs.

Makefile provided with this project supports Titan-RTX GPU, Pascal GTX 1050 GPU.
Other GPUs can be supported but Makefile needs to be changed accordingly. 


###################################SAMPLE-USAGE######################################
1. demo-BGV-PRE-Timings
	
2. demo-RGSW-PRE-Timings

$ <Program Name 1/2> 1024 16 2 1 ./demo/parametersByBits

Explainations: 
1. 1st Parameter indicates Cyclotomic number. Ring dimension is 512 = 1024/2
2. 2nd & 3rd Parameter indicates moduli. 16 is the modulus size & 2 is the CRT factor. Actual modulus size = 16 x 2 = 32.
3. 3rd Parameter is the relinearization factor or bit-decomposition base parameter.
4. 4th Parameter is the file name where parameters are stored.

  

###################################DEPENDENCIES######################################
1. To run this project, you should first install PALISADE crypto library. Make sure the path is properly described in the makefile.
