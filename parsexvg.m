function [p, arrsize] = parsexvg(pathstr)
% parsexvg: takes in terminal ls path as a list and generate data
%   use * as in terminal. for ex: "pressure_*"

files = dir(pathstr);
m = length(files);

f1 = fopen(files(1).name,'r');
temp = textscan(f1,'%f %f','Headerlines',13,'CommentStyle','@','Delimiter','\b');
fclose(f1);

arrsize = size(temp{1,2});
arrsize(2) = m;

p = zeros(arrsize);

for i = 1:m
    f1 = fopen(files(i).name, 'r');
    temp = textscan(f1,'%f %f','Headerlines',13,'CommentStyle','@','Delimiter','\b');
    p(:,i) = temp{1,2};
    fclose(f1);
end

end



