function [p, arrsize] = parsexvg(path, str)
% parsexvg: takes in terminal ls path as a list and generate data
%   use * as in terminal. for ex: "pressure_*"
if ~endsWith(path, '/')
    path = [path, '/'];
end

files = dir([path,str]);

m = length(files);

f1 = fopen([path, files(1).name],'r');
temp = textscan(f1,'%f %f','Headerlines',13,'CommentStyle','@','Delimiter','\b');
fclose(f1);

arrsize = size(temp{1,2});
arrsize(2) = m;

p = zeros(arrsize);

for i = 1:m
    f1 = fopen([path, files(i).name], 'r');
    temp = textscan(f1,'%f %f','Headerlines',13,'CommentStyle','@','Delimiter','\b');
    p(:,i) = temp{1,2};
    fclose(f1);
end

end



