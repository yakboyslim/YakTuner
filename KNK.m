function [Res_KNK] = KNK(log,igxaxis,igyaxis,logvars,bin,S50,A05,V30)

prompt = {'Maximum advance if no knock seen:','Confidence Required to confirm knock:'};
dlgtitle = 'Knock Inputs';
dims = [1 50];
definput = {'0.75','0.75'};
answer = inputdlg(prompt,dlgtitle,dims,definput)
maxadv=str2num(answer{1});
conf=1-str2num(answer{2});
map=listdlg('PromptString','Select a SP Map.','SelectionMode','single','ListString',[string(1:5) 'No map switching']);

if map==6
    currentIG{1}=zeros(16,16)
else
    if S50==1
        address=[0x27CF1A 0x27D01A 0x27D11A 0x27D21A 0x27D31A]
    elseif A05==1
        address=[0x2AFE7A 0x2AFF9A 0x2B00BA 0x2B01DA 0x2B02FA]
    elseif V30==1
        address=[0x13CF1A 0x13D01A 0x13D11A 0x13D21A 0x13D31A]
    end 
    req={address(map),16,16,95,2.666666666667,"uint8"}
    currentIG=BinRead(bin,req)
end

%% Create Derived Values

log.MAP=log.MAP.*10

%% Read Axis Values
xlabels=string(igxaxis)
ylabels=string(igyaxis)
%% Create Bins

xedges(1)=0;
xedges(length(igxaxis)+1)=inf;
for i=1:length(igxaxis)-1;
    xedges(i+1)=(igxaxis(i)+igxaxis(i+1))/2;
end

yedges(1)=0;
yedges(length(igyaxis)+1)=inf;
for i=1:length(igyaxis)-1;
    yedges(i+1)=(igyaxis(i)+igyaxis(i+1))/2;
end

%% Initialize matrixes

SUM=zeros(length(igyaxis),length(igyaxis));
COUNT=SUM;
AVG=SUM;

%% Create Trimmed Dataset


log.knkoccurred(1)=0;

allcyl=[log.KNK1 log.KNK2 log.KNK3 log.KNK4];
mincyl=min(allcyl,[],2);
outliercyl=isoutlier(allcyl);
outliercyl=outliercyl.*(allcyl)./mincyl;
outliercyl(isnan(outliercyl))=[0];
[row cyl]=find(outliercyl);
log.singlecyl(row)=cyl;
log.KNKAVG=mean(allcyl,2);

for i=2:height(log)
    if any([log.KNK1(i-1)-log.KNK1(i)>1,log.KNK2(i-1)-log.KNK2(i)>0,log.KNK3(i-1)-log.KNK3(i)>0,log.KNK4(i-1)-log.KNK4(i)>0])
        log.knkoccurred(i)=1;
    else
        log.knkoccurred(i)=0;
    end
end



%% Discretize All data

X=discretize(log.RPM,xedges);
Y=discretize(log.MAF,yedges);
log.X=X;
log.Y=Y;

columns= zeros(12,8)
rows=columns

for i=1:length(igxaxis)
    for j=1:length(igyaxis)
        temp=log;
        temp(temp.X~=i,:)=[];
        temp(temp.Y~=j,:)=[];
        tempKNK=temp;
        tempKNK(tempKNK.knkoccurred==0,:)=[];
        KR(j,i)=mean(tempKNK.KNKAVG);
        COUNT(j,i)=height(temp);
        if COUNT(j,i)>3
            ci=paramci(fitdist(temp.KNKAVG,'Normal'),'Alpha',conf);
            sigma(j,i)=ci(2,2);
            low(j,i)=ci(1,1);
            high(j,i)=ci(2,1);
            if high(j,i)<0
                AVG(j,i)=(high(j,i)+KR(j,i))/2;
            elseif high(j,i)==0 & igxaxis>2500 & igyaxis>700
                AVG(j,i)=maxadv*COUNT(j,i)/100,maxadv
            else
                AVG(j,i)=0
            end
        else
            AVG(j,i)=0;
        end
    end
end


AVG(isnan(AVG))=0
AVG(AVG>maxadv)=maxadv
inter=ceil((AVG)*5.33333333)/5.33333333
NEW=round((inter)*2.666666666666667)/2.666666666666667

resarray=NEW+currentIG{1}

Res_KNK=array2table(resarray,'VariableNames',xlabels,'RowNames',ylabels)
fig3 = uifigure('Name','IG Results',"Position",[500 500 760 360]);
uit = uitable(fig3, "Data",Res_KNK, "Position",[20 20 720 320]);
styleIndices = (KR)<0;
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'r');
addStyle(uit,s,"cell",[row,col]);

%% Plot

figure
hold on
plottemp=log
plottemp(plottemp.knkoccurred==0,:)=[]
scatter(plottemp.RPM,plottemp.MAF,abs(plottemp.KNKAVG)*100,plottemp.singlecyl)
    mycolormap = customcolormap([0 .33 .66 1], {'#0000ff','#ff0000','#00ff00','#ff00ff'});
    c = colorbar;
    c.Label.String = 'Cylinder';
    c.Ticks=[1,2,3,4]
    colormap(mycolormap);
    clim([1 4]);
    set(gca, 'Ydir', 'reverse');
    xlabel('RPM') ;
    ylabel('MAF') ;
    grid on;
    set(gca,"XAxisLocation","top");

end