function [Res_1,Res_0] = WG(log,wgxaxis,wgyaxis,currentWG0,currentWG1,logvars,plot,WGlogic)

%% Determine SWG/FF

% WGlogic = questdlg('Are you using FeedForward or SWG?','WG Logic','FF','SWG','FF');

if WGlogic==1
    log.EFF=log.RPM
    log.IFF=log.PUTSP*10
%     wgyaxis=wgyaxis/10
end

    oldwgyaxis = wgyaxis
    oldwgxaxis = wgxaxis   

%% Get other inputs

prompt = {'PUT fudge factor:','Minimum pedal (if no PUT_I_INHIBIT):','Maximum PUT delta:','Minimum Boost:'};
dlgtitle = 'WG Inputs';
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

%% Create Trimmed datasets

log=log;

if any(contains(logvars,'I_INH'))
    log(log.I_INH>0,:) = [];
else
    log(log.Pedal<minpedal,:) = [];
end

if any(contains(logvars,'DV'))
    log(log.DV>50,:) = [];
end

if any(contains(logvars,'BOOST'))
    log(log.BOOST<minboost,:) = [];
end

log(abs(log.deltaPUT)>maxdelta,:) = [];
log(log.WG_Final>98,:) = [];
log_WGopen=log;


%% Plot
if plot==true
    wait=waitbar(0,"Plotting")
    for i=1:height(log)
        pts(i,:)=[log.EFF(i),log.IFF(i),100,log.deltaPUT(i)];
        if log.VVL(i)==1
            syms(i)="^";
        else
            syms(i)="o";
        end
        waitbar(i/height(log),wait,"Choosing Points");
    end    
    hold on

    mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffff','#65c0ae','#5e4f9f'});
    c = colorbar;
    c.Label.String = 'PUT - PUT SP';
    colormap(mycolormap);
    clim([-15 15]);
    set(gca, 'Ydir', 'reverse');
    xlabel('Exh Flow Factor') ;
    ylabel('Int Flow Factor') ;
    grid on;
    set(gca,"XAxisLocation","top");
    xticks(wgxaxis);
    yticks(wgyaxis);

    for i=1:height(log)
        if mod(i,ceil(height(log)/1000))==0
            scatr=scatter(pts(i,1),pts(i,2),pts(i,3),pts(i,4),syms(i));
        end
        if mod(i,ceil(height(log)/100))==0
            waitbar(i/height(log),wait,"Plotting Points"); 
        end
    end
    close(wait)
    
    figedit = uifigure('Name','DO NOT CLOSE THIS TABLE USING THE X, USE CONTINUE BUTTON',"Position",[500 500 760 320]);
    lbl2 = uilabel(figedit,"Text",'WG X Axis','Position',[20 290 720 30]);
    uit2 = uitable(figedit, "Data",wgxaxis, "Position",[20 210 720 80], 'ColumnEditable',true);
    lbl3 = uilabel(figedit,"Text",'WG Y Axis','Position',[20 150 720 30]);
    uit3 = uitable(figedit, "Data",wgyaxis, "Position",[20 70 720 80], 'ColumnEditable',true);
    btn1 = uicontrol(figedit,'String','Replot','Position',[20 30 150 30])
    btn1.Callback={@plotButtonPushed,uit2,uit3}
    c = uicontrol(figedit,'String','Continue','Position',[220 30 150 30])
    c.Callback={@cb_continue,figedit}

    uiwait(figedit)
%     hold off
    oldwgyaxis = wgyaxis
    oldwgxaxis = wgxaxis   
    wgxaxis = uit2.Data
    wgyaxis = uit3.Data
    close(figedit)
end

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


exhlabels=string(wgxaxis);
intlabels=string(wgyaxis);

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

blend1=(columns1+rows1)/200

for i=1:length(wgxaxis)
    for j=1:length(wgyaxis)
        temp=log_VVL1;
        temp(temp.X~=i,:)=[];
        temp(temp.Y~=j,:)=[];
        AVGtemp=mean(temp.WGNEED)
        COUNT(j,i)=height(temp)
        current(j,i)=interp2(oldwgxaxis,oldwgyaxis,currentWG1,wgxaxis(i),wgyaxis(j),"linear",0)
            if COUNT(j,i)>3
            ci=paramci(fitdist(temp.WGNEED,'Normal'),'Alpha',0.5)
            sigma(j,i)=ci(2,2)
            low(j,i)=ci(1,1)
            high(j,i)=ci(2,1)
%             AVGtemp=0
            if isnan(current(j,i))
                AVG1(j,i)=(blend1(j,i)+AVGtemp/100)/2
            elseif low(j,i)>current(j,i)
                AVG1(j,i)=(blend1(j,i)+AVGtemp/100)/2
            elseif high(j,i)<current(j,i)
                AVG1(j,i)=(blend1(j,i)+AVGtemp/100)/2
            else
                AVG1(j,i)=current(j,i)
            end
        else
            AVG1(j,i)=current(j,i)
        end
    end
end

AVG1=round(AVG1*16384)/16384

Data1=current-AVG1
Data1=Data1./Data1

Res_1=array2table(AVG1,'VariableNames',exhlabels,'RowNames',intlabels);

%% Discretize VVL0

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

blend0=(columns0+rows0)/200

for i=1:length(wgxaxis)
    for j=1:length(wgyaxis)
        temp=log_VVL0;
        temp(temp.X~=i,:)=[];
        temp(temp.Y~=j,:)=[];
        AVGtemp=mean(temp.WGNEED)
        COUNT(j,i)=height(temp)
        current(j,i)=interp2(oldwgxaxis,oldwgyaxis,currentWG0,wgxaxis(i),wgyaxis(j))
            if COUNT(j,i)>3
            ci=paramci(fitdist(temp.WGNEED,'Normal'),'Alpha',0.5)
            sigma(j,i)=ci(2,2)
            low(j,i)=ci(1,1)
            high(j,i)=ci(2,1)
%             AVGtemp=0
            if isnan(current(j,i))
                AVG0(j,i)=(blend0(j,i)+AVGtemp/100)/2                
            elseif low(j,i)>current(j,i)
                AVG0(j,i)=(blend0(j,i)+AVGtemp/100)/2
            elseif high(j,i)<current(j,i)
                AVG0(j,i)=(blend0(j,i)+AVGtemp/100)/2
            else
                AVG0(j,i)=current(j,i)
            end
        else
            AVG0(j,i)=current(j,i)
        end
    end
end

AVG0=round(AVG0*16384)/16384

Data0=current-AVG0
Data0=Data0./Data0

Res_0=array2table(AVG0,'VariableNames',exhlabels,'RowNames',intlabels);


% %% Plot Scatters
% if plot == 1
%     f1=tiledlayout(2,1)
%     nexttile
%     hold on
%     for i=1:height(log)
%         if log.VVL(i)==1
%             if log.WG_Final(i)>98
%                 c1=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"^","filled");
%             else
%                 o1=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"^");
%             end
%         else
%             if log.WG_Final(i)>98
%                 c0=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"o","filled");
%             else
%                 o0=scatter(log.EFF(i),log.IFF(i),100,log.deltaPUT(i),"o");
%             end
%         end
%     end
%     
%     mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffff','#65c0ae','#5e4f9f'});
%     c = colorbar;
%     c.Label.String = 'PUT - PUT SP';
%     colormap(mycolormap);
%     clim([-15 15]);
%     set(gca, 'Ydir', 'reverse');
%     xlabel('Exh Flow Factor') ;
%     ylabel('Int Flow Factor') ;
%     % set(gca,'TickLength',[1 1])
%     grid on;
%     set(gca,"XAxisLocation","top");
%     xticks(wgxaxis);
%     yticks(wgyaxis);
%     rows=[1,1,1,1]
%     rowsfound=[find(log.RPM>2000,1) find(log.RPM>3000,1) find(log.RPM>4000,1) find(log.RPM>5000,1)];
%     rows(1,1:numel(rowsfound)) = rowsfound;
%     lscatter(log.EFF(rows),log.IFF(rows),[2000 3000 4000 5000]);
%     hold off
%     
%     nexttile
%     hold on
%     for i=1:height(log)
%         if log.VVL(i)==1
%             if log.WG_Final(i)>98
%                 c1p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"^","filled");
%             else
%                 o1p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"^");
%             end
%         else
%             if log.WG_Final(i)>98
%                 c0p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"o","filled");
%             else
%                 o0p=scatter(log.EFF(i),log.IFF(i),100,log.WGCL(i),"o");
%             end
%         end
%     end
%     
%     mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffff','#65c0ae','#5e4f9f'});
%     c = colorbar;
%     c.Label.String = 'WG CL';
%     colormap(mycolormap);
%     clim([-10 10]);
%     set(gca, 'Ydir', 'reverse');
%     xlabel('Exh Flow Factor') ;
%     ylabel('Int Flow Factor') ;
%     % set(gca,'TickLength',[1 1])
%     grid on;
%     set(gca,"XAxisLocation","top");
%     xticks(wgxaxis);
%     yticks(wgyaxis);
%     rows=[1,1,1,1]
%     rowsfound=[find(log.RPM>2000,1) find(log.RPM>3000,1) find(log.RPM>4000,1) find(log.RPM>5000,1)];
%     rows(1,1:numel(rowsfound)) = rowsfound;
%     lscatter(log.EFF(rows),log.IFF(rows),[2000 3000 4000 5000]);
%     hold off
% end
% 

%% Display Tables

fig0 = uifigure('Name','VVL0 WG Results',"Position",[500 500 760 360]);
uit = uitable(fig0, "Data",Res_0, "Position",[20 20 720 320]);

styleIndices = ~ismissing(Data0);
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'gr');
addStyle(uit,s,"cell",[row,col]);

fig1 = uifigure('Name','VVL1 WG Results',"Position",[500 500 760 360]);
uit = uitable(fig1, "Data",Res_1, "Position",[20 20 720 320]);

styleIndices = ~ismissing(Data1);
[row,col] = find(styleIndices);
s = uistyle("BackgroundColor",'gr');
addStyle(uit,s,"cell",[row,col]);
end

function plotButtonPushed(src,event,uit2,uit3)
    xticks(uit2.Data)
    yticks(uit3.Data)
end

function cb_continue(~,~,figedit)
    uiresume(figedit);
end