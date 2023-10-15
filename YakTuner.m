%% Read Log
clear all
close all

%% Get Variable list

varconv=readtable(fullfile(getcurrentdir,"variables.csv"),Delimiter=',',TextType='string',ReadVariableNames=false);

wgxaxis=csvread(fullfile(getcurrentdir,"x_axis.csv"));
wgyaxis=csvread(fullfile(getcurrentdir,"y_axis.csv"));

fig = uifigure('Name','DO NOT CLOSE THIS TABLE USING THE X, USE CONTINUE BUTTON',"Position",[500 500 400 200]);
cbx1 = uicheckbox(fig,"Text","Tune WG?",'Position',[20 160 100 30]);
cbx2 = uicheckbox(fig,"Text","Plot WG? Adds considerable time for large logs",'Position',[40 130 300 30]);
cbx3 = uicheckbox(fig,"Text","Tune MAF?",'Position',[140 160 100 30]);
cbx4 = uicheckbox(fig,"Text","Edit variable conversion and axis tables?",'Position',[20 100 350 30]);
cbx5 = uicheckbox(fig,"Text","Output results to CSV?",'Position',[20 70 350 30]);
c = uicontrol(fig,'String','CONTINUE','Callback','uiresume(fig)')
uiwait(fig)
WGtune = cbx1.Value
MAFtune = cbx3.Value
plot = cbx2.Value
edit = cbx4.Value
save = cbx5.Value
close(fig)

% edit = questdlg('Edit variable conversion and axis tables?','Edit?','Yes','No','Yes');
if edit == 1
    figedit = uifigure('Name','DO NOT CLOSE THIS TABLE USING THE X, USE CONTINUE BUTTON',"Position",[500 500 760 500]);
    lbl1 = uilabel(figedit,"Text",'Variables','Position',[20 450 720 30]);
    uit1 = uitable(figedit, "Data",varconv, "Position",[20 350 720 100], 'ColumnEditable',true);
    lbl2 = uilabel(figedit,"Text",'WG X Axis','Position',[20 290 720 30]);
    uit2 = uitable(figedit, "Data",wgxaxis, "Position",[20 210 720 80], 'ColumnEditable',true);
    lbl3 = uilabel(figedit,"Text",'WG Y Axis','Position',[20 150 720 30]);
    uit3 = uitable(figedit, "Data",wgyaxis, "Position",[20 70 720 80], 'ColumnEditable',true);
    c = uicontrol(figedit,'String','CONTINUE','Callback','uiresume(figedit)')
    uiwait(figedit)    
    varconv = uit1.Data;
    wgxaxis = uit2.Data;
    wgyaxis = uit3.Data;
    writetable(varconv,fullfile(getcurrentdir,"variables.csv"),'WriteVariableNames',false)
    writematrix(wgxaxis,fullfile(getcurrentdir,"x_axis.csv"))
    writematrix(wgyaxis,fullfile(getcurrentdir,"y_axis.csv"))
    close(figedit)
end



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

%% Convert Variables
logvars = log.Properties.VariableNames;
for i=2:width(varconv)
    if any(contains(logvars,varconv(1,i)))
        log=renamevars(log,varconv(1,i),varconv(2,i));
    end
end

logvars = log.Properties.VariableNames;

if WGtune == 1
    [Res_1,Res_0] = WG(log,wgxaxis,wgyaxis,logvars,plot)
    if save ==1
        writetable(Res_1,fullfile(pathopen,"VVL1 Results.csv"),'WriteRowNames',true);
        writetable(Res_0,fullfile(pathopen,"VVL0 Results.csv"),'WriteRowNames',true);
    end
end

if MAFtune == 1
    Res_MAF = MAF(log,logvars)
    if save == 1
        writetable(Res_MAF,fullfile(pathopen,"MAF_STD Results.csv"),'WriteRowNames',true);
    end
end








function [Res_1,Res_0] = WG(log,wgxaxis,wgyaxis,logvars,plot)

exhlabels=string(wgxaxis);
intlabels=string(wgyaxis);

%% Determine SWG/FF

WGlogic = questdlg('Are you using FeedForward or SWG?','WG Logic','FF','SWG','FF');

if strcmp(WGlogic,'SWG')
    log.EFF=log.RPM
    log.IFF=log.PUTSP
    wgyaxis=wgyaxis/10
end

%% Get other inputs

prompt = {'PUT fudge factor:','Minimum pedal:','Maximum PUT delta:','Minimum boost:'};
dlgtitle = 'Inputs';
dims = [1 50];
definput = {'0.71','50','10','0'};
answer = inputdlg(prompt,dlgtitle,dims,definput)
fudge=str2num(answer{1});
minpedal=str2num(answer{2});
maxdelta=str2num(answer{3});
minboost=str2num(answer{4});

%% Create Derived Values
log.deltaPUT=log.PUT-log.PUTSP;
log.WGNEED=log.WG_Final-log.deltaPUT.*fudge;
log.WGCL=log.WG_Final-log.WG_Base;


%% Create Bins

wgxedges(1)=wgxaxis(1);
wgxedges(length(wgxaxis)+1)=wgxaxis(length(wgxaxis))+2;
for i=1:length(wgxaxis)-1;
    wgxedges(i+1)=(wgxaxis(i)+wgxaxis(i+1))/2;
end

wgyedges(1)=wgyaxis(1);
wgyedges(length(wgyaxis)+1)=wgyaxis(length(wgyaxis))+2;
for i=1:length(wgyaxis)-1;
    wgyedges(i+1)=(wgyaxis(i)+wgyaxis(i+1))/2;
end


%% Create Trimmed datasets

log=log;
if any(contains(logvars,'I_INH'))
    log(log.I_INH>0,:) = [];
else
    log(log.Pedal<minpedal,:) = [];
end

log(log.DV>50,:) = [];
log(log.BOOST<minboost,:) = [];
log(abs(log.deltaPUT)>maxdelta,:) = [];
log_WGopen=log;
log_WGopen(log_WGopen.WG_Final>98,:) = [];

log_WGopen.X=discretize(log_WGopen.EFF,wgxedges);
log_WGopen.Y=discretize(log_WGopen.IFF,wgyedges);



log_VVL1=log_WGopen;
log_VVL1(log_VVL1.VVL~=1,:) = [];
log_VVL0=log_WGopen;
log_VVL0(log_VVL0.VVL~=0,:) = [];



%% Initialize matrixes

SUM1=zeros(length(wgyaxis),length(wgxaxis));
COUNT1=SUM1;
SUM0=SUM1;
COUNT0=SUM1;
COUNTFAC=zeros(4,4);
SUMFAC=zeros(4,4);
COUNTOFS=zeros(4,4);
SUMOFS=zeros(4,4);
columns1= zeros(10,16)
rows1=columns1
rows0=rows1
columns0=columns1


%% Discretize VVL1

for i=1:height(log_VVL1)
   weight=abs(log_VVL1.deltaPUT(i))*(0.5-abs(log_VVL1.EFF(i)-log_VVL1.X(i)))*(0.5-abs(log_VVL1.IFF(i)-log_VVL1.Y(i)));
   SUM1(log_VVL1.Y(i),log_VVL1.X(i))=SUM1(log_VVL1.Y(i),log_VVL1.X(i))+weight*log_VVL1.WGNEED(i);
   COUNT1(log_VVL1.Y(i),log_VVL1.X(i))=COUNT1(log_VVL1.Y(i),log_VVL1.X(i))+1;
end

for i=1:length(wgxaxis)
        temp=log_VVL1;
        temp(temp.X~=i,:)=[];
        if height(temp)>1
            columns1(:,i) = lsq_lut_piecewise( temp.IFF, temp.WGNEED, wgyaxis )
        end
end

for j=1:length(wgyaxis)
        temp=log_VVL1;
        temp(temp.Y~=j,:)=[];
        if height(temp)>1
            rows1(j,:) = lsq_lut_piecewise( temp.EFF, temp.WGNEED, wgxaxis )
        end
end
data1=COUNT1>0
AVG1=round(((SUM1./COUNT1)+rows1.*data1+columns1.*data1)/3)/100;
Res_1=array2table(AVG1,'VariableNames',exhlabels,'RowNames',intlabels);

%% Discretize VVL0

for i=1:height(log_VVL0)
   weight=abs(log_VVL0.deltaPUT(i))*(0.5-abs(log_VVL0.EFF(i)-log_VVL0.X(i)))*(0.5-abs(log_VVL0.IFF(i)-log_VVL0.Y(i)));
   SUM0(log_VVL0.Y(i),log_VVL0.X(i))=SUM0(log_VVL0.Y(i),log_VVL0.X(i))+(weight)*log_VVL0.WGNEED(i);
   COUNT0(log_VVL0.Y(i),log_VVL0.X(i))=COUNT0(log_VVL0.Y(i),log_VVL0.X(i))+weight;
end

for i=1:length(wgxaxis)
        temp=log_VVL0;
        temp(temp.X~=i,:)=[];
        if height(temp)>1
            columns0(:,i) = lsq_lut_piecewise( temp.IFF, temp.WGNEED, wgyaxis)
        end
end

for j=1:length(wgyaxis)
        temp=log_VVL0;
        temp(temp.Y~=j,:)=[];
        if height(temp)>1   
            rows0(j,:) = lsq_lut_piecewise( temp.EFF, temp.WGNEED, wgxaxis )  
        end
end
data0=COUNT0>0
AVG0=round(((SUM0./COUNT0)+rows0.*data0+columns0.*data0)/3)/100;
Res_0=array2table(AVG0,'VariableNames',exhlabels,'RowNames',intlabels);


%% Plot Scatters
if plot == 1
    f1=tiledlayout(2,1)
    nexttile
    hold on
    for i=1:height(log)
        if log.VVL(i)==1
            if log.WG_Final(i)>98
                c1=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"^","filled");
            else
                o1=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"^");
            end
        else
            if log.WG_Final(i)>98
                c0=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"o","filled");
            else
                o0=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"o");
            end
        end
    end
    
    mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffff','#65c0ae','#5e4f9f'});
    c = colorbar;
    c.Label.String = 'PUT - PUT SP';
    colormap(mycolormap);
    clim([-15 15]);
    set(gca, 'Ydir', 'reverse');
    xlabel('Exh Flow Factor') ;
    ylabel('Int Flow Factor') ;
    % set(gca,'TickLength',[1 1])
    grid on;
    set(gca,"XAxisLocation","top");
    xticks(wgxaxis);
    yticks(wgyaxis);
    rows=[1,1,1,1]
    rowsfound=[find(log.RPM>2000,1) find(log.RPM>3000,1) find(log.RPM>4000,1) find(log.RPM>5000,1)];
    rows(1,1:numel(rowsfound)) = rowsfound;
    lscatter(log.EFF(rows),log.IFF(rows),[2000 3000 4000 5000]);
    hold off
    
    nexttile
    hold on
    for i=1:height(log)
        if log.VVL(i)==1
            if log.WG_Final(i)>98
                c1p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"^","filled");
            else
                o1p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"^");
            end
        else
            if log.WG_Final(i)>98
                c0p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"o","filled");
            else
                o0p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"o");
            end
        end
    end
    
    mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffff','#65c0ae','#5e4f9f'});
    c = colorbar;
    c.Label.String = 'WG CL';
    colormap(mycolormap);
    clim([-10 10]);
    set(gca, 'Ydir', 'reverse');
    xlabel('Exh Flow Factor') ;
    ylabel('Int Flow Factor') ;
    % set(gca,'TickLength',[1 1])
    grid on;
    set(gca,"XAxisLocation","top");
    xticks(wgxaxis);
    yticks(wgyaxis);
    rows=[1,1,1,1]
    rowsfound=[find(log.RPM>2000,1) find(log.RPM>3000,1) find(log.RPM>4000,1) find(log.RPM>5000,1)];
    rows(1,1:numel(rowsfound)) = rowsfound;
    lscatter(log.EFF(rows),log.IFF(rows),[2000 3000 4000 5000]);
    hold off
end

fig0 = uifigure('Name','VVL0 WG Results',"Position",[500 500 760 360]);
uit = uitable(fig0, "Data",Res_0, "Position",[20 20 720 320]);

styleIndices = ~ismissing(Res_0);
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'gr');
addStyle(uit,s,"cell",[row,col]);

fig1 = uifigure('Name','VVL1 WG Results',"Position",[500 500 760 360]);
uit = uitable(fig1, "Data",Res_1, "Position",[20 20 720 320]);

styleIndices = ~ismissing(Res_1);
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'gr');
addStyle(uit,s,"cell",[row,col]);
end

function [Res_MAF] = MAF(log,logvars)
[filestock,pathstock]=uigetfile('*.csv','Select Stock MAF_STD Table CSV File');
current=csvread(fullfile(pathstock,filestock))

prompt = {'Maximum confidence:','Minimum count to discard:','Count for max confidence:'};
dlgtitle = 'Inputs';
dims = [1 50];
definput = {'0.75','10','100'};
answer = inputdlg(prompt,dlgtitle,dims,definput)
fudge=str2num(answer{1});
mincount=str2num(answer{2});
maxcount=str2num(answer{3});


%% Create Derived Values

log.MAP=log.MAP.*10
log.LAM_PCT=100*(log.LAMBDA-log.LAMBDA_SP)./log.LAMBDA_SP
log.ADD_MAF=log.STFT+log.LTFT+log.MAF_COR
 
log(log.EngState<2,:) = [];
log(log.EngState>4,:) = [];
%% Read Axis Values
% stock=csvread(fullfile(getcurrentdir,"VE_stock.csv"))
xaxis=current(1,2:width(current))
yaxis=reshape(current(2:height(current),1),1,[])
xlabels=string(xaxis)
ylabels=string(yaxis)
current(1,:)=[]
current(:,1)=[]
%% Create Bins

xedges(1)=0;
xedges(length(xaxis)+1)=inf;
for i=1:length(xaxis)-1;
    xedges(i+1)=(xaxis(i)+xaxis(i+1))/2;
end

yedges(1)=0;
yedges(length(yaxis)+1)=inf;
for i=1:length(yaxis)-1;
    yedges(i+1)=(yaxis(i)+yaxis(i+1))/2;
end

%% Initialize matrixes

SUM=zeros(length(yaxis),length(xaxis));
COUNT=SUM;
AVG=SUM;

%% Discretize VVL1

X=discretize(log.RPM,xedges);
Y=discretize(log.MAP,yedges);
log.X=X;
log.Y=Y;

columns= zeros(12,8)
rows=columns

for i=1:length(xaxis)
        temp=log;
        temp(temp.X~=i,:)=[];
        columns(:,i) = lsq_lut_piecewise( temp.MAP, temp.ADD_MAF, yaxis )
end

for j=1:length(yaxis)
        temp=log;
        temp(temp.Y~=j,:)=[];
        rows(j,:) = lsq_lut_piecewise( temp.RPM, temp.ADD_MAF, xaxis )          
end

blend=(columns+rows)/2

for i=1:length(xaxis)
    for j=1:length(yaxis)
        temp=log;
        temp(temp.X~=i,:)=[];
        temp(temp.Y~=j,:)=[];
        AVGtemp=mean(temp.ADD_MAF)
        COUNT(j,i)=height(temp)
        if COUNT(j,i)>3
            ci=paramci(fitdist(temp.ADD_MAF,'Normal'),'Alpha',0.1)
            sigma(j,i)=ci(2,2)
            low(j,i)=ci(1,1)
            high(j,i)=ci(2,1)
            if low(j,i)>current(j,i)
                AVG(j,i)=blend(j,i)
            elseif high(j,i)<current(j,i)
                AVG(j,i)=blend(j,i)
            else
                AVG(j,i)=current(j,i)
            end
        else
            AVG(j,i)=current(j,i)
        end
    end
end

COUNT(COUNT>maxcount)=[maxcount]
COUNT=COUNT/maxcount

COUNT(isnan(COUNT))=0
AVG(isnan(AVG))=0
CHANGE=(AVG-current).*COUNT*fudge
NEW=round((current+CHANGE)*5.12,0)/5.12

Res_MAF=array2table(NEW,'VariableNames',xlabels,'RowNames',ylabels)
fig3 = uifigure('Name','MAF Results',"Position",[500 500 760 360]);
uit = uitable(fig3, "Data",Res_MAF, "Position",[20 20 720 320]);
styleIndices = (CHANGE);
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'gr');
addStyle(uit,s,"cell",[row,col]);
end