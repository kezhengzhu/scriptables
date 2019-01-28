N = 2500001;
acf_imp2 = zeros(N,1);
% fprintf('tau = 0000000');
% for tau = 0:N-1
%     fprintf('\b\b\b\b\b\b\b%7d',tau);
%     shifted = p(tau+1:end);
%     acf_imp2(tau+1) = (p(1:end-tau)'*shifted)/(N-tau);
% end

fprintf('tau = 0000000');
for tau = 0:(N-1)
    fprintf('\b\b\b\b\b\b\b%7d',tau);
    cs = 0;
    for t0 = 1:(N-tau)
        cs = cs + p(t0)*p(t0+tau);
    end
    acf_imp(tau+1) = cs/(N-tau);
end

