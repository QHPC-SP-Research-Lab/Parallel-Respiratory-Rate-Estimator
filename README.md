# Noise-tolerant NMF-based Parallel Algorithm for Respiratory Rate Estimation

## Overview

This project presents a parallel driver designed to address the challenges of respiratory rate (RR) estimation in real-world environments, combining multi-core architectures with parallel and high-performance techniques. Our proposed system employs a non-negative matrix factorization (NMF) approach to mitigate the impact of noise interference in the input signal. This NMF approach is guided by pre-trained bases of respiratory sounds and incorporates an orthogonal constraint to enhance accuracy. The proposed solution is tailored for real-time processing on low-power hardware, demonstrating promising outcomes in terms of accuracy and computational efficiency.

## Implementation

Our proposed solution builds upon the principles of NMF and parallel computing. The parallel driver efficiently handles the challenges posed by noisy environments, leveraging multi-core architectures and high-performance techniques. The NMF-based algorithm incorporates pre-trained bases of respiratory sounds and an orthogonal constraint to enhance accuracy. The implementation is based on a combination of C and Python and is optimized for real-time processing on low-power hardware, ensuring practical applicability in diverse healthcare settings.

## Getting Started

To utilize our noise-tolerant NMF-based parallel algorithm for respiratory rate estimation, follow these steps:

1. Clone the repository
2. Refer to the following instructions to set up dependencies
3. Execute the provided scripts for a comprehensive understanding of the library's usage

## Prerequisites

Before you proceed with the installation, ensure that you have the following prerequisites:

- **C Compiler:** Install a C compiler on your system. If you haven't installed one yet, you can use the GNU Compiler Collection (GCC) for compatibility.

- **BLAS/LAPACK Library:** Obtain and install a distribution of the BLAS/LAPACK library. This library is crucial for performing efficient numerical linear algebra operations.

- **FFTW Library:** Download and install the FFTW library, which is essential for fast Fourier transform computations. Ensure proper configuration for optimal performance.

- **Python 3:** Make sure you have Python 3 installed on your system. This is required for running the project's Python script.

## Compilation

Follow these steps to compile the C code:

1. Open your terminal using a bash-compatible shell.

2. Run the following commands:
   
   ``make``

Ensure that you properly link against the BLAS, LAPACK, and FFTW libraries, as well as the math library during the compilation process.

## Execution

To run the project, execute the compiled binary along with the Python script. Follow these steps:

1. Open your terminal using a bash-compatible shell.

2. Run the following commands:

	``python RRmain.py filename.wav number_iterations``

Replace **``filename.wav``** with the path to your input audio file and **``number_iterations``** with the desired number of iterations for the algorithm.

Ensure that the input audio files are in monochannel format and use a 16-bit per sample resolution for compatibility with the project.

## Contributing

We welcome contributions to enhance the capabilities of our noise-tolerant NMF-based parallel algorithm for respiratory rate estimation. Fork the repository, make improvements, and submit a pull request to the master branch.

## Reference

If you use this algorithm in your research, please cite the following paper:

> P. Revuelta-Sanz, A.J. Muñoz-Montoro, J. Torre-Cruz, F.J. Canadas-Quesada, J. Ranilla. "Noise-tolerant NMF-based parallel algorithm for respiratory rate estimation", 2023.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

For inquiries or further information, please contact José Ranilla at ranilla@uniovi.es.
