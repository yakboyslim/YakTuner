%% Read Log
clear all
close all

%% Get Variable list

varconv=readtable(fullfile(getcurrentdir,"variables.csv"),Delimiter=',',TextType='string',ReadVariableNames=false);

% wgxaxis=csvread(fullfile(getcurrentdir,"x_axis.csv"));
% wgyaxis=csvread(fullfile(getcurrentdir,"y_axis.csv"));

fig = uifigure('Name','DO NOT CLOSE THIS TABLE USING THE X, USE CONTINUE BUTTON',"Position",[500 500 400 340]);
cbx1 = uicheckbox(fig,"Text","Tune WG?",'Position',[20 160 100 30]);
cbx2 = uicheckbox(fig,"Text","Edit WG Axis? ",'Position',[40 130 100 30]);
cbx6 = uicheckbox(fig,"Text","SWG? ",'Position',[160 130 100 30]);
cbx3 = uicheckbox(fig,"Text","Tune MAF?",'Position',[140 160 100 30]);
cbx7 = uicheckbox(fig,"Text","S50",'Position',[20 100 100 30]);
cbx8 = uicheckbox(fig,"Text","A05",'Position',[140 100 100 30]);
cbx5 = uicheckbox(fig,"Text","Output results to CSV?",'Position',[20 70 350 30]);
lbl1 = uilabel(fig,"Text",'Variables','Position',[20 300 360 30]);
uit1 = uitable(fig, "Data",varconv, "Position",[20 200 360 100], 'ColumnEditable',true);
c = uicontrol(fig,'String','CONTINUE','Callback','uiresume(fig)') 
uiwait(fig)
WGtune = cbx1.Value
MAFtune = cbx3.Value
plot = cbx2.Value
save = cbx5.Value
WGlogic = cbx6.Value
S50 = cbx7.Value
A05 = cbx8.Value
varconv = uit1.Data;
writetable(varconv,fullfile(getcurrentdir,"variables.csv"),'WriteVariableNames',false)
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
        log=outerjoin(log,openlog,'MergeKeys', true);
    end
end

[filebin,pathbin]=uigetfile('*.bin','Select Bin File');
bin=fopen(fullfile(pathbin,filebin));

if S50==1
    address=[0x24B669 0x24B62A 0x24BB66 0x21974A 0x2196FC 0x2199F6 0x219B36]
elseif A05==1
    address=[0x277EDF 0x277EA0 0x27837C 0x23D0E0 0x23D092]
else
    errordlg('Must select S50 or A05')
end
rows=[12 1 1 1 1 10 10]
cols=[8 12 8 10 16 16 16]
offset=[128 0 0 0 0 32768 32768]
res=[5.12 12.06 1 16384 16384 16384 16384]
if WGlogic==1
    req(4:6)=1
end
prec=["uint8" "uint16" "uint16" "uint16" "uint16" "uint16" "uint16"]
req={address rows cols offset res prec}

output = BinRead(bin,req)
wgxaxis=output{5}
wgyaxis=output{4}
mafxaxis=output{3}
mafyaxis=output{2}
currentMAF=output{1}
currentWG0=output{6}
currentWG1=output{7}

%% Convert Variables
logvars = log.Properties.VariableNames;
for i=2:width(varconv)
    if any(contains(logvars,varconv(1,i)))
        log=renamevars(log,varconv(1,i),varconv(2,i));
    end
end

logvars = log.Properties.VariableNames;

if WGtune == 1
    [Res_1,Res_0] = WG(log,wgxaxis,wgyaxis,currentWG0,currentWG1,logvars,plot,WGlogic)
    if save ==1
        writetable(Res_1,fullfile(pathopen,"VVL1 Results.csv"),'WriteRowNames',true);
        writetable(Res_0,fullfile(pathopen,"VVL0 Results.csv"),'WriteRowNames',true);
    end
end

if MAFtune == 1
    Res_MAF = MAF(log,mafxaxis,mafyaxis,currentMAF,logvars)
    if save == 1
        writetable(Res_MAF,fullfile(pathopen,"MAF_STD Results.csv"),'WriteRowNames',true);
    end
end

