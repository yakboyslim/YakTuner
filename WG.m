function WG(log,wgxaxis,wgyaxis,logvars)
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