function [x, umie] = calcmie(sigma, epsi, r, a, nint, cutoff)
if nargin < 5
    nint = 500;
end
if nargin < 6
    cutoff = 5;
end

x = (cutoff*sigma/(nint)):(cutoff*sigma/(nint)):cutoff*sigma;
cmie = r/(r-a) * r/a ^ (a/(r-a));
sr = sigma./x;
umie = (cmie * epsi) .* (sr.^r - sr.^a);
end