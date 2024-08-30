# Overview
This project simulates a single carrier communication system, focusing on various modulation schemes such as BPSK, QPSK, 8PSK, BFSK, and 16QAM. The goal is to model the system, analyze its performance in an AWGN channel, and calculate the Bit Error Rate (BER) for different modulation schemes.

# Project Objectives
## 1. Modulation Schemes:

- Simulate the performance of BPSK, QPSK, 8PSK, BFSK, and 16QAM modulation schemes.
- Generate and analyze constellation diagrams for each modulation scheme.
## 2. Channel Modeling:

- Model the communication channel as an Additive White Gaussian Noise (AWGN) channel using MATLAB's randn function to add noise to the transmitted signal.
## 3. Demapping and BER Calculation:

- Implement a demapper to decode the received symbols and compare the output bit stream to the input bit stream.
- Calculate and plot BER vs. Eb/No for each modulation scheme.
- Compare the simulated BER with the theoretical BER or a tight upper bound for each modulation scheme.
## 4. Analysis of BFSK:

- Analyze the Basis Frequency Shift Keying (BFSK) signal.
- Derive the basis functions and the baseband equivalent signals.
- Simulate and plot the Power Spectral Density (PSD) of the BFSK signal.
# Implementation Details
## 1. System Model
- The system model includes three main components: Mapper, Channel, and Demapper.
- The mapper converts input data bits into symbols using the selected modulation scheme.
- The channel introduces AWGN to simulate real-world transmission conditions.
- The demapper recovers the transmitted symbols and calculates the BER.
## 2. Tasks
- BER Curves: Simulate and plot BER vs. Eb/No for BPSK, QPSK, 8PSK, and 16QAM. Compare these results with theoretical BER curves.
- QPSK Comparison: Compare the BER performance of two different QPSK constellations.
- BFSK Analysis: Perform a detailed analysis of BFSK, including the derivation of basis functions, baseband equivalent signal expressions, and PSD simulation.
# Project Structure
- Scripts/: MATLAB scripts for performing simulations, plotting results, and analyzing the system.
- Results/: Generated plots and output data from the simulations.
