function [output] = BinRead(bin,req)
% [fileopen,pathopen]=uigetfile()
% bin=fopen(fullfile(pathopen,fileopen));

for j=1:length(req{1})
fseek(bin,req{1}(j),'bof')
temp=fread(bin,[req{2}(j) req{3}(j)],req{6}(j))
output{j}=(temp-req{4}(j))./req{5}(j)
end