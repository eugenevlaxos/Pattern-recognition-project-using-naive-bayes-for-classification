%%lowpass fir using hamming window

b = fir1(40,0.3*pi);
figure(1);
plot(b);
title('Lowpass Filter');
xlabel('n (samples)');
ylabel('Amplitude');

figure(2);
impz(b);

figure(3);
stepz(b);

fvtool(b);

figure(5);
grpdelay(b);

Wp = 0.1;
Ws = 0.3;
Rp = 1;
Rs = 40;
[N,Wn] = buttord(Wp,Ws,Rp,Rs);
[numerator,denominator] = butter(N,Wn);

Fs = 1;
[num,den] = bilinear(numerator,denominator,Fs);

figure(6);
impz(num,den);

figure(7);
stepz(num,den);

fvtool(num,den);

figure(9);
grpdelay(num,den);

figure(10);
zplane(num,den);

%B1
w1 = pi*mod(80/105,1);
w2 = mod(w1 + pi/4,pi);

limits = [0, pi];

L = 16;
X = rectwin(L);

n = 0:L-1;
signal = cos(w1 * n) + 0.5 * cos(w2*n);

signal = X .* signal;

N = L;
Xk = fft(signal,N);
figure(11);
plot(abs(Xk));
xlim(limits);

N = 2^14;
Xk = fft(signal,N);
figure(12);
plot(abs(Xk));
xlim(limits);

L = 64;
X = rectwin(L);

n = 0:L-1;
signal = cos(w1 * n) + 0.5 * cos(w2*n);

signal = X .* signal;


N = L;
Xk = fft(signal,N);
figure(13);
plot(abs(Xk));
xlim(limits);

N = 2^14;
Xk = fft(signal,N);
figure(14);
plot(abs(Xk));
xlim(limits);

L = 512;
X = rectwin(L);

n = 0:L-1;
signal = cos(w1 * n) + 0.5 * cos(w2*n);

signal = X .* signal;


N = L;
Xk = fft(signal,N);
figure(15);
plot(abs(Xk));
xlim(limits);

N = 2^14;
Xk = fft(signal,N);
figure(16);
plot(abs(Xk));
xlim(limits);

w1 = pi*mod(80/105,1);
w2 = mod(w1 + pi/4,pi);

w = (w1 + w2)/2;

L = 64;
X = rectwin(L);

n = 0:L-1;
signal = cos(w * n) + 0.5 * cos(w*n);

signal = X .* signal;

N = L;
Xk = fft(signal,N);
figure(17);
plot(abs(Xk));
xlim(limits);

N = 2^14;
Xk = fft(signal,N);
figure(18);
plot(abs(Xk));
xlim(limits);

r = audiorecorder(22000,16,1);
recordblocking(r,5);
p = play(r);
myspeech = getaudiodata(r);

figure(19);
spectrogram(myspeech,hamming(220),110);

figure(20);
spectrogram(myspeech,hamming(2200),110);

filtered_data = filter(num,den,myspeech);
figure(21);
spectrogram(filtered_data,hamming(220),110);

numerator = 1;
denominator = [1,(1:79) * 0,-0.9999];
figure(22);
[H,T] = impz(numerator,denominator);
plot(H);

figure(23);
spectrogram(H);
figure(24);
zplane(numerator,denominator);

numerator = [1,-0.96];
denominator = 1;

figure(25);
[H,T] = impz(numerator,denominator);
plot(H);

figure(26);
spectrogram(H);

figure(27);
zplane(numerator,denominator);
