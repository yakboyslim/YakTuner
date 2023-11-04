function [results] = MAF(log,mafxaxis,mafyaxis,maftables,combmodes,logvars)
% [filestock,pathstock]=uigetfile('*.csv','Select Stock MAF_STD Table CSV File');
% current=csvread(fullfile(pathstock,filestock))

prompt = {'Confidence required to make change:'};
dlgtitle = 'MAF Correction Inputs';
dims = [1 50];
definput = {'0.95'};
answer = inputdlg(prompt,dlgtitle,dims,definput)
conf=str2num(answer{1});
maxcount=100


%% Create Derived Values

log.MAP=log.MAP.*10

if ~any(contains(logvars,'LAM_DIF'))
    log.LAM_DIF=1./log.LAMBDA_SP-1./log.LAMBDA
end

if any(contains(logvars,'FAC_LAM_OUT'))
    log.ADD_MAF=log.FAC_LAM_OUT-log.LAM_DIF 
else
    log.ADD_MAF=log.STFT+log.LTFT-log.LAM_DIF 
end

if any(contains(logvars,'MAF_COR'))
    log.ADD_MAF=log.ADD_MAF+log.MAF_COR
end

% log(log.EngState<2,:) = [];
% log(log.EngState>4,:) = [];
log(log.state_lam~=1,:) = [];
% log(log.LAMBDA==0,:) = [];


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



%% Discretize MAF

X=discretize(log.RPM,xedges);
Y=discretize(log.MAP,yedges);
log.X=X;
log.Y=Y;

%% Initialize matrixes

for IDX=0:3

    SUM=zeros(length(mafyaxis),length(mafxaxis));
    COUNT=SUM;
    AVG=SUM;
    columns= zeros(12,8)
    rows=columns
    current=maftables{IDX+1}
    IDXmodes=find(combmodes==IDX)-1

    if any(contains(logvars,'MAF_COR'))
        test=current
    else
        test=zeros(length(mafyaxis),length(mafxaxis))
    end

    temp1=log
    diffcmb=setdiff(log.CMB,IDXmodes)
    for k=1:length(diffcmb)
        temp1(temp1.CMB==diffcmb(k),:)=[]
    end

    for i=1:length(mafxaxis)
        temp=temp1;
        temp(temp.X~=i,:)=[];
        tempcol = lsq_lut_piecewise( temp.MAP, temp.ADD_MAF, mafyaxis )
        if size(tempcol)==[8,1]
            columns(:,i) = tempcol
        end
    end
    
    for j=1:length(mafyaxis)
        temp=temp1;
        temp(temp.Y~=j,:)=[];
        temprow = lsq_lut_piecewise( temp.RPM, temp.ADD_MAF, mafxaxis )
        if size(temprow)==[8,1]
            rows(j,:) = temprow
        end
    end
    
    blend=(columns+rows)/2
    interpfac=.25
    
    
    for i=1:length(mafxaxis)
        for j=1:length(mafyaxis)
            temp=temp1;
            temp(temp.X~=i,:)=[];
            temp(temp.Y~=j,:)=[];
            AVGtemp=mean(temp.ADD_MAF)
            COUNT(j,i)=height(temp)
            if COUNT(j,i)>3
                ci=paramci(fitdist(temp.ADD_MAF,'Normal'),'Alpha',conf)
                sigma(j,i)=ci(2,2)
                low(j,i)=ci(1,1)
                high(j,i)=ci(2,1)
                if low(j,i)>test(j,i)
                    AVG(j,i)=(blend(j,i)*interpfac+low(j,i)*(1-interpfac))
                elseif high(j,i)<test(j,i)
                    AVG(j,i)=(blend(j,i)*interpfac+high(j,i)*(1-interpfac))
                    else
                    AVG(j,i)=test(j,i)
                end
            else
                AVG(j,i)=test(j,i)
            end
        end
    end
    
    COUNT(COUNT>maxcount)=[maxcount]
    COUNT=COUNT/maxcount
    
    COUNT(isnan(COUNT))=0
    AVG(isnan(AVG))=0
    CHANGE=(AVG-test).*COUNT
    NEW=round((current+CHANGE)*5.12,0)/5.12
    
    my_field = strcat('IDX',num2str(IDX))
    results.(my_field)=array2table(NEW,'VariableNames',xlabels,'RowNames',ylabels)
    fig3 = uifigure('Name',strcat('MAF Results IDX []: ',num2str(IDX)),"Position",[500 500 760 360]);
    uit = uitable(fig3, "Data",results.(my_field), "Position",[20 20 720 320]);
    styleIndices = (CHANGE);
    [row,col] = find(styleIndices);
    s = uistyle("BackgroundColor",'gr');
    addStyle(uit,s,"cell",[row,col]);
end

end