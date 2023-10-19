function [Res_MAF] = MAF(log,mafxaxis,mafyaxis,current,logvars)
% [filestock,pathstock]=uigetfile('*.csv','Select Stock MAF_STD Table CSV File');
% current=csvread(fullfile(pathstock,filestock))

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
% mafxaxis=current(1,2:width(current))
% mafyaxis=reshape(current(2:height(current),1),1,[])
xlabels=string(mafxaxis)
ylabels=string(mafyaxis)
% current(1,:)=[]
% current(:,1)=[]
%% Create Bins

xedges(1)=0;
xedges(length(mafxaxis)+1)=inf;
for i=1:length(mafxaxis)-1;
    xedges(i+1)=(mafxaxis(i)+mafxaxis(i+1))/2;
end

yedges(1)=0;
yedges(length(mafyaxis)+1)=inf;
for i=1:length(mafyaxis)-1;
    yedges(i+1)=(mafyaxis(i)+mafyaxis(i+1))/2;
end

%% Initialize matrixes

SUM=zeros(length(mafyaxis),length(mafxaxis));
COUNT=SUM;
AVG=SUM;

%% Discretize VVL1

X=discretize(log.RPM,xedges);
Y=discretize(log.MAP,yedges);
log.X=X;
log.Y=Y;

columns= zeros(12,8)
rows=columns

for i=1:length(mafxaxis)
        temp=log;
        temp(temp.X~=i,:)=[];
        columns(:,i) = lsq_lut_piecewise( temp.MAP, temp.ADD_MAF, mafyaxis )
end

for j=1:length(mafyaxis)
        temp=log;
        temp(temp.Y~=j,:)=[];
        rows(j,:) = lsq_lut_piecewise( temp.RPM, temp.ADD_MAF, mafxaxis )          
end

blend=(columns+rows)/2

for i=1:length(mafxaxis)
    for j=1:length(mafyaxis)
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