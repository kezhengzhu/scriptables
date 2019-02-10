function [acf, arrsize] = fftacf(p)

% Using p as dataset: no. of dataset = no of columns, each column = 1
% dataset

arrsize = size(p);
trun = size(p,1);
cols = size(p,2);
p2 = vertcat(p, zeros(arrsize));

% Complex matrix, squared modulus
X = abs(fft(p2)).^2;

p2 = ifft(X);
acf = p2(1:trun,:);
tsteps = 0:(trun-1);
tsteps = tsteps';

% normalisation
for i = 1:cols
    acf(:,i) = acf(:,i) ./ (trun - tsteps);
end

end