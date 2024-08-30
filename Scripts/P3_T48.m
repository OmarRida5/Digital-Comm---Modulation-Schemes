close all
clear
%% PART 1 - DIFFERENT MODULATION SCHEMES

% generate bits
n_bits = 24*10000; %choosing n_bits to be divisible by: 2,3,4
bits = randi([0 1], 1,n_bits);

symbols_BPSK  = mapper_BPSK(bits);
symbols_QPSK_gray  = mapper_QPSK(bits);
symbols_QPSK_not_gray = mapper_QPSK_not_gray(bits);
symbols_8PSK  = mapper_8PSK(bits);
symbols_BFSK  = mapper_BFSK(bits);
symbols_16QAM = mapper_16QAM(bits);

SNR_dB = -4:14; % Eb/No, Eb=1

% allocating the BER array for different modulation schemes
BER_BPSK  = zeros(size(SNR_dB));
BER_BFSK  = zeros(size(SNR_dB));
BER_QPSK  = zeros(size(SNR_dB));
BER_8PSK  = zeros(size(SNR_dB));
BER_16QAM = zeros(size(SNR_dB));
BER_QPSK_not_gray = zeros(size(SNR_dB));

%% AWGN channel
for k=SNR_dB
    SNR_linear = 10^(k/10);

    % calculating energy per bit
    Eb_BPSK_BFSK_QPSK = 1;
    Eb_8PSK = (1/3);
    Eb_16QAM = 10/4;

    % for Eb = 1 --> No = 1/SNR_linear --> Ns = Eb_for_each_modulation*(No)
    Ns_BPSK_BFSK_QPSK = Eb_BPSK_BFSK_QPSK/SNR_linear;
    Ns_8PSK  = Eb_8PSK/SNR_linear;
    Ns_16QAM = Eb_16QAM/SNR_linear;

    % add the noise to he transmitted signal to model the additive white gaussian noise
    received_symbol_BPSK = sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_BPSK)) + 1i*sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_BPSK));
    received_symbol_BPSK = received_symbol_BPSK + symbols_BPSK;

    received_symbol_BFSK = sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_BFSK)) + 1i*sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_BFSK));
    received_symbol_BFSK = received_symbol_BFSK + symbols_BFSK;

    received_symbol_QPSK_gray = sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_QPSK_gray)) + 1i*sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_QPSK_gray));
    received_symbol_QPSK_gray = received_symbol_QPSK_gray + symbols_QPSK_gray;

    received_symbol_QPSK_not_gray = sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_QPSK_gray)) + 1i*sqrt(Ns_BPSK_BFSK_QPSK/2)*randn(size(symbols_QPSK_gray));
    received_symbol_QPSK_not_gray = received_symbol_QPSK_not_gray + symbols_QPSK_not_gray;


    received_symbol_8PSK = sqrt(Ns_8PSK/2)*randn(size(symbols_8PSK)) + 1i*sqrt(Ns_8PSK/2)*randn(size(symbols_8PSK));
    received_symbol_8PSK = received_symbol_8PSK + symbols_8PSK;

    received_symbol_16QAM = sqrt(Ns_16QAM/2)*randn(size(symbols_16QAM)) + 1i*sqrt(Ns_16QAM/2)*randn(size(symbols_16QAM));
    received_symbol_16QAM = received_symbol_16QAM + symbols_16QAM;

    %% Demapper
    demapped_BPSK          = demapper_BPSK(received_symbol_BPSK);
    demapped_BFSK          = demapper_BFSK(received_symbol_BFSK);
    demapped_8PSK          = demapper_8PSK(received_symbol_8PSK);
    demapped_QPSK          = demapper_QPSK(received_symbol_QPSK_gray);
    demapped_QPSK_not_gray = demapper_QPSK_not_gray(received_symbol_QPSK_not_gray);
    demapped_16QAM = demapper_16QAM(received_symbol_16QAM);

    %% BER calculations
    num_errors_BPSK = count_errors(demapped_BPSK,bits);
    BER_BPSK(k+5) = num_errors_BPSK/n_bits;

    num_errors_BFSK = count_errors(demapped_BFSK,bits);
    BER_BFSK(k+5) = num_errors_BFSK/n_bits;

    num_errors_8PSK = count_errors(demapped_8PSK,bits);
    BER_8PSK(k+5) = num_errors_8PSK/n_bits;

    num_errors_QPSK = count_errors(demapped_QPSK,bits);
    BER_QPSK(k+5) = num_errors_QPSK/n_bits;

    num_errors_QPSK_not_gray = count_errors(demapped_QPSK_not_gray,bits);
    BER_QPSK_not_gray(k+5) = num_errors_QPSK_not_gray/n_bits;

    num_errors_16QAM = count_errors(demapped_16QAM,bits);
    BER_16QAM(k+5) = num_errors_16QAM/n_bits;
end

%% calculation of the theoretical values:
theoretical_8PSK_BER = (1/3)*erfc(sqrt(3*10.^(SNR_dB/10))*sin(pi/8));
theoretical_BPSK_QPSK_BER = 0.5*erfc(sqrt(10.^(SNR_dB/10)));
theoretical_16QAM_BER = (3/4)*2/4*erfc(sqrt(10.^(SNR_dB/10)/2.5));
theoretical_BFSK_BER = (1/2)*erfc(sqrt(10.^(SNR_dB/10)/2));

%% Plot BER for different modulation schemes
figure;
% Plot BPSK
semilogy(SNR_dB, BER_BPSK, '-o', 'DisplayName', 'BPSK', 'Color', [0.8500, 0.3250, 0.0980]);
hold on;
% Plot BFSK
semilogy(SNR_dB, BER_BFSK, '-s', 'DisplayName', 'BFSK', 'Color', [0, 0.4470, 0.7410]);
% Plot QPSK
semilogy(SNR_dB, BER_QPSK, '-^', 'DisplayName', 'QPSK', 'Color', [0.9290, 0.6940, 0.1250]);
% Plot 8PSK
semilogy(SNR_dB, BER_8PSK, '-d', 'DisplayName', '8PSK', 'Color', [0.4940, 0.1840, 0.5560]);
% Plot 16QAM
semilogy(SNR_dB, BER_16QAM, '-+', 'DisplayName', '16QAM', 'Color', [0.4660, 0.6740, 0.1880]);
% Plot theoretical BER for 8PSK with the same color as practical BER
semilogy(SNR_dB, theoretical_8PSK_BER, '--', 'DisplayName', '8PSK theoretical', 'Color', [0.4940, 0.1840, 0.5560]);
% Plot theoretical BER for BPSK and QPSK with the same color as practical BER
semilogy(SNR_dB, theoretical_BPSK_QPSK_BER, '--', 'DisplayName', 'QPSK theoretical', 'Color', [0.9290, 0.6940, 0.1250]);
% Plot theoretical BER for BFSK with the same color as practical BER
semilogy(SNR_dB, theoretical_BFSK_BER, '--', 'DisplayName', 'BFSK theoretical', 'Color', [0, 0.4470, 0.7410]);
% Plot theoretical BER for 16QAM with the same color as practical BER
semilogy(SNR_dB, theoretical_16QAM_BER, '--', 'DisplayName', '16QAM theoretical', 'Color', [0.4660, 0.6740, 0.1880]);
% Limit y-axis until 10^-5
ylim([10^-5, inf]);
hold off;
grid on;
xlabel('Eb/No (dB)');
ylabel('Bit Error Rate (BER)');
title('Bit Error Rate (BER) for Different Modulation Schemes');
legend('Location', 'best');

figure
semilogy(SNR_dB, BER_QPSK_not_gray, '-^', 'DisplayName', 'QPSK (not Gray)');
hold on
semilogy(SNR_dB, BER_QPSK, '-^', 'DisplayName', 'QPSK (Gray)');
% Limit y-axis until 10^-5
ylim([10^-5, inf]);
hold off;
grid on;
xlabel('Eb/No (dB)');
ylabel('Bit Error Rate (BER)');
title('Effect of Gray coding on BER');
legend('Location', 'best');

figure
semilogy(SNR_dB, BER_BFSK, '-^', 'DisplayName', 'BFSK simulated');
hold on
semilogy(SNR_dB, theoretical_BFSK_BER, '-^', 'DisplayName', 'BFSK theoretical');
% Limit y-axis until 10^-5
ylim([10^-5, inf]);
hold off;
grid on;
xlabel('Eb/No (dB)');
ylabel('Bit Error Rate (BER)');
title('BER for BFSK');
legend('Location', 'best');


%% PART 2 - BFSK baseband

%% data generation
% BFSK %
% bit '0' --> mapped to s1_bb = sqrt(2/Tb)
% bit '1' --> mapped to s2_bb = sqrt(2/Tb)*cos(2*pi*delta_f*t)
clear
% bit specification
Tb = 1;
n_samples_per_bit = 7;
Ts = Tb/n_samples_per_bit;
Fs = 1/Ts;
% realizations specification
ensem_number = 15000;
n_bits_per_realiz = 100;
n_samples_per_realiz = n_bits_per_realiz*n_samples_per_bit;

% each row represents new realization
raw_bits = randi([0,1], ensem_number, n_bits_per_realiz+1); % another bit will be added to perform delay then rotate (later this bit must be removed)
symbols_BFSK_bb = zeros(ensem_number,n_samples_per_realiz+n_samples_per_bit);
symbols_BFSK_bb_delayed = zeros(ensem_number,n_samples_per_realiz+n_samples_per_bit);
for i=1:ensem_number
    symbols_BFSK_bb(i,:) = mapper_BFSK_bb(raw_bits(i,:),Tb,Ts);
end

% perform random delay on each realization
for i = 1:ensem_number
    delay = randi([1, n_samples_per_bit-1]);
    symbols_BFSK_bb_delayed(i,:) = circshift(symbols_BFSK_bb(i,:), delay);
end

% cut the last added bit
symbols_BFSK_bb_delayed = symbols_BFSK_bb_delayed(:, 1:end-n_samples_per_bit);

%% autocorrelation
autocorrelation=zeros([1 n_samples_per_realiz]);
tau_range_full = -(n_samples_per_realiz-1):(n_samples_per_realiz-1);
tau_range_part = -(50):(50);

% Number of runs
num_runs = 10;
% Initialize accumulator
psd_accumulator = zeros(1, 1399); % Assuming N is the length of your PSD
for run = 1:num_runs
    % Your code for generating bit stream and computing autocorrelation
    % and PSD for each run goes here
    for tau=1:n_samples_per_realiz
        for i=1:ensem_number
            autocorrelation(tau)=autocorrelation(tau)+(conj(symbols_BFSK_bb_delayed(i,1)).*symbols_BFSK_bb_delayed(i,tau));
        end
        autocorrelation(tau)=autocorrelation(tau)/ensem_number;
    end

    %create double sided autocorrelaion
    left_side_part=autocorrelation(end:-1:2);
    double_sided_autocorrelation=[(left_side_part),conj(autocorrelation)];
    double_sided_autocorrelation_part = double_sided_autocorrelation(n_samples_per_realiz-50:n_samples_per_realiz+50);
    
    % Compute the FFT
    N = length(double_sided_autocorrelation);                % Length of the signal
    frequencies = (-Fs/2:Fs/N:Fs/2-Fs/N);                    % Frequency axis for FFT (centered at zero)   

    PSD = fftshift(fft(double_sided_autocorrelation))/N;  % Compute FFT, shift, and normalize    
    
    % Accumulate PSD values
    psd_accumulator = psd_accumulator + PSD;
end

% Average the PSD
average_psd = psd_accumulator / num_runs;


% Plot
figure;
subplot(2,1,1);
plot(tau_range_full,abs(double_sided_autocorrelation));
title("Statistical Autocorrelation");
xlabel('Tau (sec)');
ylabel('Autocorrelation');

subplot(2,1,2);
plot(tau_range_part,abs(double_sided_autocorrelation_part));
title("Statistical Autocorrelation zoomed");
xlabel('Tau (sec)');
ylabel('Autocorrelation');

%% PSD of base band BFSK
% Compute the FFT
% N = length(double_sided_autocorrelation);                % Length of the signal
% frequencies = (-Fs/2:Fs/N:Fs/2-Fs/N);         % Frequency axis for FFT (centered at zero)
% PSD = fftshift(fft(double_sided_autocorrelation))/N;  % Compute FFT, shift, and normalize

% Plot the FFT
figure;
subplot(1,2,1);
plot(frequencies, abs(average_psd));
title('PSD of BFSK Base Band signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(1,2,2);
plot(frequencies, abs(average_psd));
ylim([0 0.05]);
title('PSD of BFSK Base Band signal zoomed');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

%% functions

%% MAPPERS
function symbols = mapper_BPSK(bits)
symbols = 2*bits -1;
end

function symbols = mapper_QPSK(bits)
symbols = (2*bits(1:2:end-1) -1) + 1i*((2*bits(2:2:end) -1));
end

function symbols = mapper_QPSK_not_gray(bits)
reshaped_bits = reshape(bits, 2, [])';
mapped_bits = [1 0; 0 1; 0 0; 1 1];
corresponding_symbol = [1+1i -1+1i -1-1i 1-1i];
[~, index] = ismember(reshaped_bits, mapped_bits, "rows");
symbols = corresponding_symbol(index);
end


function symbols = mapper_8PSK(bits)
Gray_mapped_bits = [0 0 0; 0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 1 1; 1 0 1; 1 0 0];
reshaped_bits = reshape(bits, 3, [])';
[~, indices] = ismember(reshaped_bits, Gray_mapped_bits, "rows");
symbols = exp(1i*2*(indices'-1)*pi/8);
end

function symbols = mapper_BFSK(bits)
% mapp 0 --> 1
% mapp 1 --> i
symbols = (1*abs(bits-1) + 1i*(bits));
end

function symbols = mapper_16QAM(bits)
reshaped_bits = reshape(bits, 2, [])';
Gray_mapped_bits = [0 0; 0 1; 1 1; 1 0];
% determining the in phase and quadrature coefficients:
[~, index] = ismember(reshaped_bits, Gray_mapped_bits, "rows");
% in-phase is determined by the first index (first 2 bits) while the quad
% coef is determined by the second index(second 2 bits)
in_phase_coeff = -3 + 2*(index(1:2:end-1)'-1);
quad_coeff = -3 + 2*(index(2:2:end)'-1);
symbols = in_phase_coeff + 1i*quad_coeff;
end


%% DEMAPPERS
function bits = demapper_BPSK(symbols)
% Map positive elements to 1 and negative elements to 0
bits = ones(size(symbols));
bits(symbols < 0) = 0; % Update elements where array is negative
end

function bits = demapper_BFSK(symbols)
% if in-phase coeff > quad coeff --> demapp to 0
% else --> de mapp to 1
bits = zeros(size(symbols));
bits(real(symbols) < imag(symbols)) = 1;
end

function bits = demapper_8PSK(symbols)
ref_coordinates = exp(1i*2*(0:7)*pi/8);
demapped_bits = [0 0 0; 0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 1 1; 1 0 1; 1 0 0];
bits = zeros(1, numel(symbols) * size(demapped_bits, 2));
for i = 1:numel(symbols)
    x = symbols(i);
    % Calculate the distance between x and each constellation point
    distances = abs(x - ref_coordinates);
    [~, closest_index] = min(distances);
    % Assign the corresponding demapped bits
    start_idx = (i - 1) * size(demapped_bits, 2) + 1;
    end_idx = i * size(demapped_bits, 2);
    bits(start_idx:end_idx) = demapped_bits(closest_index, :);
end
end

function bits = demapper_QPSK(symbols)
% each of the in-phase and quad components are cosidered as individual BPSK
% demapp the in-phase component
in_phase_bits = ones(size(symbols));
in_phase_bits(real(symbols) < 0) = 0; % Update elements where array is negative
% demapp the quad component
quad_bits = ones(size(symbols));
quad_bits(imag(symbols) < 0) = 0; % Update elements where array is negative
% Combine in-phase and quadrature components to form the bits
bits = reshape([in_phase_bits; quad_bits], 1, []);
end

function bits = demapper_QPSK_not_gray(symbols)
demapped_bits = [1 0; 0 1; 0 0; 1 1];
corresponding_symbol = [1+1i -1+1i -1-1i 1-1i];
bits = zeros(1, numel(symbols) * size(demapped_bits, 2));
for i = 1:numel(symbols)
    x = symbols(i);
    % Calculate the distance between x and each constellation point
    distances = abs(x - corresponding_symbol);
    [~, closest_index] = min(distances);
    % Assign the corresponding demapped bits
    start_idx = (i - 1) * size(demapped_bits, 2) + 1;
    end_idx = i * size(demapped_bits, 2);
    bits(start_idx:end_idx) = demapped_bits(closest_index, :);
end

end


function bits = demapper_16QAM(symbols)
% Demap the in-phase component
in_phase_bits = ones(numel(symbols), 2);

indices = find(real(symbols) < -2);
subset_size = numel(indices);
in_phase_bits(indices, :) = repmat([0 0], subset_size, 1);

indices = find(-2<=real(symbols)&real(symbols)<0);
subset_size = numel(indices);
in_phase_bits(indices, :) = repmat([0 1], subset_size, 1);

indices = find(0<=real(symbols)&real(symbols)<2);
subset_size = numel(indices);
in_phase_bits(indices, :) = repmat([1 1], subset_size, 1);

indices = find(2<= real(symbols));
subset_size = numel(indices);
in_phase_bits(indices, :) = repmat([1 0], subset_size, 1);

% Demap the quad component
quad_bits = ones(numel(symbols), 2);
indices = find(imag(symbols) < -2);
subset_size = numel(indices);
quad_bits(indices, :) = repmat([0 0], subset_size, 1);

indices = find(-2<=imag(symbols)&imag(symbols)<0);
subset_size = numel(indices);
quad_bits(indices, :) = repmat([0 1], subset_size, 1);

indices = find(0<=imag(symbols)&imag(symbols)<2);
subset_size = numel(indices);
quad_bits(indices, :) = repmat([1 1], subset_size, 1);

indices = find(2<= imag(symbols));
subset_size = numel(indices);
quad_bits(indices, :) = repmat([1 0], subset_size, 1);

bits_2D = [in_phase_bits, quad_bits];
bits = reshape(bits_2D',[1 size(bits_2D,1)*size(bits_2D,2)]);
end

function num_errors = count_errors(bits, received_bits)
% Count the number of errors between symbols and received symbols
num_errors = sum(bits ~= received_bits);
end

%% Function BFSK baseband mapper
function symbols = mapper_BFSK_bb(bits,Tb,Ts)
% create the time axis for one bit
t_one_bit = 0:Ts:Tb-Ts;
delta_f = 1/Tb;
Eb = 1;
% define the base band equivalent functions of BFSK
s1_BB = sqrt(2*Eb/Tb)*ones(size(t_one_bit));
s2_BB = sqrt(2*Eb/Tb)*cos(2*pi*delta_f*t_one_bit) + 1j*sqrt(2/Tb)*sin(2*pi*delta_f*t_one_bit);

symbols_2D = ones(numel(bits), numel(t_one_bit));

% map 0s to s1_bb and 1s to s2_bb
indices = find(bits == 0);
subset_size = numel(indices);
symbols_2D(indices, :) = repmat(s1_BB, subset_size, 1);

indices = find(bits == 1);
subset_size = numel(indices);
symbols_2D(indices, :) = repmat(s2_BB, subset_size, 1);

symbols = reshape(symbols_2D',[1 size(symbols_2D,1)*size(symbols_2D,2)]);
end

