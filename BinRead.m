function [output] = BinRead(bin,req)
% [fileopen,pathopen]=uigetfile()
% bin=fopen(fullfile(pathopen,fileopen));

% req1=[0x24B669 0x24B62A 0x24BB66]
% req2=[12 1 1]
% req3=[8 12 8]
% req4=[128 0 0]
% req5=[5.12 12.06 1]
% req6=["uint8" "uint16" "uint16"]

for j=1:length(req{1})
fseek(bin,req{1}(j),'bof')
temp=fread(bin,[req{2}(j) req{3}(j)],req{6}(j))
output{j}=(temp-req{4}(j))./req{5}(j)
end

% req=[0x24B669 12 8 128 5.12]
% fseek(bin,req(1),'bof')
% MAF_STD=fread(bin,[req(2) req(3)])
% MAF_STD=(MAF_STD-req(4))./req(5)