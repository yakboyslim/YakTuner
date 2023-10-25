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
% log.LAM_PCT=100*(log.LAMBDA-log.LAMBDA_SP)./log.LAMBDA_SP
% log.LAM_DIF=1./log.LAMBDA_SP-1./log.LAMBDA
log.ADD_MAF=log.FAC_LAM_COR+log.MAF_COR-log.LAM_DIF
 
log(log.EngState<2,:) = [];
log(log.EngState>4,:) = [];
log(log.state_lam~=1,:) = [];
log(log.LAMBDA==0,:) = [];


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
    temp1=log
    diffcmb=setdiff(log.CMB,IDXmodes)
    for k=1:length(diffcmb)
        temp1(temp1.CMB==diffcmb(k),:)=[]
    end

    for i=1:length(mafxaxis)
            temp=temp1;
            temp(temp.X~=i,:)=[];
            columns(:,i) = lsq_lut_piecewise( temp.MAP, temp.ADD_MAF, mafyaxis )
    end
    
    for j=1:length(mafyaxis)
            temp=temp1;
            temp(temp.Y~=j,:)=[];
            rows(j,:) = lsq_lut_piecewise( temp.RPM, temp.ADD_MAF, mafxaxis )          
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
                if low(j,i)>current(j,i)
                    AVG(j,i)=(blend(j,i)*interpfac+low(j,i)*(1-interpfac))/2
                elseif high(j,i)<current(j,i)
                    AVG(j,i)=(blend(j,i)*interpfac+high(j,i)*(1-interpfac))/2
    
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
    CHANGE=(AVG-current).*COUNT
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