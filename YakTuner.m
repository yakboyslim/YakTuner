%% Read Log
clear all
close all

%% Get Variable list

varconv=readtable(fullfile(getcurrentdir,"variables.csv"),Delimiter=',',TextType='string',ReadVariableNames=false);

% wgxaxis=csvread(fullfile(getcurrentdir,"x_axis.csv"));
% wgyaxis=csvread(fullfile(getcurrentdir,"y_axis.csv"));

fig = uifigure('Name','YakTuner',"Position",[500 500 400 200]);
cbx1 = uicheckbox(fig,"Text","Tune WG?",'Position',[20 160 100 30]);
cbx2 = uicheckbox(fig,"Text","Edit WG Axis? ",'Position',[40 130 100 30]);
cbx6 = uicheckbox(fig,"Text","SWG? ",'Position',[160 130 100 30]);
cbx3 = uicheckbox(fig,"Text","Tune MAF?",'Position',[140 160 100 30]);
cbx9 = uicheckbox(fig,"Text","Tune Ignition?",'Position',[260 160 100 30]);
cbx7 = uicheckbox(fig,"Text","S50",'Position',[20 100 100 30],'Value',1);
cbx8 = uicheckbox(fig,"Text","A05",'Position',[140 100 100 30]);
cbx11 = uicheckbox(fig,"Text","V30",'Position',[260 100 100 30]);
cbx5 = uicheckbox(fig,"Text","Output results to CSV?",'Position',[20 70 350 30]);
cbx10 = uicheckbox(fig,"Text","Reset Variables?",'Position',[260 70 350 30]);

c = uicontrol(fig,'String','CONTINUE','Callback','uiresume(fig)') 
uiwait(fig)
WGtune = cbx1.Value
MAFtune = cbx3.Value
IGtune = cbx9.Value
plot = cbx2.Value
save = cbx5.Value
WGlogic = cbx6.Value
S50 = cbx7.Value
A05 = cbx8.Value
V30 = cbx11.Value
var_reset = cbx10.Value
% varconv = uit1.Data;
% writetable(varconv,fullfile(getcurrentdir,"variables.csv"),'WriteVariableNames',false)
close(fig)

varconv=table2array(varconv);


%% Open Files

[fileopen,pathopen]=uigetfile('*.csv','Select Log CSV File','MultiSelect','on');

if iscell(fileopen)
    log=readtable(fullfile(pathopen,string(fileopen(1))),"VariableNamingRule","preserve");
    log=log(:,1:width(log)-1);
else
    log=readtable(fullfile(pathopen,fileopen),"VariableNamingRule","preserve");
    log=log(:,1:width(log)-1);
end
if iscell(fileopen)
    for i=2:length(fileopen)
        openlog=readtable(fullfile(pathopen,string(fileopen(i))),"VariableNamingRule","preserve");
        openlog=openlog(:,1:width(openlog)-1);
%         log=outerjoin(log,openlog,'MergeKeys', true);
        log=[log;openlog];
    end
end

[filebin,pathbin]=uigetfile('*.bin','Select Bin File');
bin=fopen(fullfile(pathbin,filebin));

%% Parse bin

if S50==1
    address=[0x24B642 0x24B62A 0x24BB66 0x21974A 0x2196FC 0x2199F6 0x219B36 0x23CDBC 0x23CE5A]
elseif A05==1
    address=[0x277EB8 0x277EA0 0x27837C 0x23D0E0 0x23D092 0x23D368 0x23D4A8 0x267A3E 0x267ADE]
elseif V30==1
    address=[0x22D926 0x22D90E 0x22DE5C 0x212E76 0x212E28 0x213134 0x213274 0x230634 0x2306D2]
else
    errordlg('Must select S50 or A05')
end
rows=[1 1 1 1 1 10 10 1 1]
cols=[39 12 8 10 16 16 16 16 16]
offset=[0 0 0 0 0 32768 32768 0 0]
res=[1 12.06 1 16384 16384 16384 16384 23.59071274298056 1]
if WGlogic==1
    res(4)=1/.082917524986648
    res(5)=1
end
prec=["uint8" "uint16" "uint16" "uint16" "uint16" "uint16" "uint16" "uint16" "uint16"]
req={address rows cols offset res prec}

output = BinRead(bin,req)
combmodes=output{1}
mafyaxis=output{2}
mafxaxis=output{3}
wgyaxis=output{4}
wgxaxis=output{5}
currentWG0=output{6}
currentWG1=output{7}
igyaxis=output{8}
igxaxis=output{9}


if S50==1
    address=[0x24B669 0x24B6C9 0x24B729 0x24B789]
elseif A05==1
    address=[0x277EDF 0x277F3F 0x277F9F 0x277FFF]
elseif V30==1
    address=[0x22D94D 0x22D9AD 0x22DA0D 0x22DA6D]
else
    errordlg('Must select S50, A05, or V30')
end
rows=[12 12 12 12]
cols=[8 8 8 8]
offset=[128 128 128 128]
res=[5.12 5.12 5.12 5.12]
prec=["uint8" "uint8" "uint8" "uint8"]
mafreq={address rows cols offset res prec}
maftables = BinRead(bin,mafreq)

%% Convert Variables
logvars = log.Properties.VariableNames;

missingvars=[]

if var_reset==1
    missingvars=[2:width(varconv)]
else
    for i=2:width(varconv)
        if any(contains(logvars,varconv(1,i)))
            log=renamevars(log,varconv(1,i),varconv(2,i));
        else
            missingvars(end+1) = i;
        end
    end
end

if isempty(missingvars)==0
    for i=1:length(missingvars)
        pick=listdlg('PromptString',{"Select a variable for:  ",varconv(3,missingvars(i)),""},'SelectionMode','single','ListString',cat(2,varconv(1,missingvars(i)),string(logvars)));
        if pick>1
            varconv(1,missingvars(i))=logvars(pick-1)
        end
        log=renamevars(log,varconv(1,missingvars(i)),varconv(2,missingvars(i)))
    end
    varconv=array2table(varconv)
    writetable(varconv,fullfile(getcurrentdir,"variables.csv"),'WriteVariableNames',false)
end

logvars = log.Properties.VariableNames;


%% Tunes

if WGtune == 1
    [Res_1,Res_0] = WG(log,wgxaxis,wgyaxis,currentWG0,currentWG1,logvars,plot,WGlogic)
    if save ==1
        writetable(Res_1,fullfile(pathopen,"VVL1 Results.csv"),'WriteRowNames',true);
        writetable(Res_0,fullfile(pathopen,"VVL0 Results.csv"),'WriteRowNames',true);
    end
end

if MAFtune == 1
    [MAFresults] = MAF(log,mafxaxis,mafyaxis,maftables,combmodes,logvars)
    if save == 1
        writetable(MAFresults.("IDX0"),fullfile(pathopen,"MAF_STD[0] Results.csv"),'WriteRowNames',true);
        writetable(MAFresults.("IDX1"),fullfile(pathopen,"MAF_STD[1] Results.csv"),'WriteRowNames',true);
        writetable(MAFresults.("IDX2"),fullfile(pathopen,"MAF_STD[2] Results.csv"),'WriteRowNames',true);
        writetable(MAFresults.("IDX3"),fullfile(pathopen,"MAF_STD[3] Results.csv"),'WriteRowNames',true);
    end
end

if IGtune == 1
    Res_KNK = KNK(log,igxaxis,igyaxis,logvars,bin,S50,A05,V30)
    if save == 1
        writetable(Res_KNK,fullfile(pathopen,"KNK Results.csv"),'WriteRowNames',true);
    end
end