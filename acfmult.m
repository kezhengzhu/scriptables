function [acf, arrsize] = acfmult(y,lags)
% acfmult calculates y(t)*y(t+tau) instead of (y(t) - ybar) * (y(t+tau) -
% ybar)
[rows, cols] = size(y);
if rows < cols
    y = y';
    [rows, cols] = size(y);
end
N = rows;
if lags > N
    error("No. of lags must be less than size of data (no. of rows)");
end

arrsize = [lags+1, cols];
acf = zeros(arrsize);

for col = 1:cols
    fprintf('Dataset col = %3d; tau = 0000000', col);
    for tau = 0:lags
        fprintf('\b\b\b\b\b\b\b%7d',tau);
        cs = 0;
        for t0 = 1:(N-tau)
            cs = cs + y(t0,col)*y(t0+tau,col);
        end
        acf(tau+1,col) = cs/(N-tau);
    end
    fprintf('\n');
end

end

