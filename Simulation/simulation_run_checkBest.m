clear all;
close all;

Top = 20;
%MU_list = 10^(-5) * [  20 15 10 5 1 ]; % Num of immagrants Large to small
MU_list =  10*10^(-5 ); % Num of immagrants Large to small
%MU_list =  [ 8*10^(-5 ) 10^(-4 ) 5*10^(-4 ) 10^(-3 ) 5*10^(-3) 10^(-2)] ; % Num of immagrants Large to small

K0_list = [ 0.990 ];    % Num of offsprings (Less and equal than 1)

%W_list = 5 * 10.^ [-4 -2 0 2 4  ];   % Time period between offspring 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%W_list = [ 10.^-5 5*10.^-5 10.^-4 5*10.^-4 10.^-3 10.^-2 ];   % Time period between offspring
%W_list = 10*10.^-5   % Time period between offspring
%W_list = [  5*10.^-5 10.^-4 1.5*10.^-4 3*10.^-4 ] ;
%W_list = [ 7*10.^-5 10.^-4 3*10.^-4 5*10.^-4 7*10.^-4 ] ;
W_list = 10.^-4;
%W_list = [ 5*1*10.^-3 1*10.^-2 1*10.^-1 ] ;
%W_list = [ 10.^-5 5*10.^-5 10.^-4 ] ;
%W_list = [10.^-6 10.^-5 5*10.^-5 10.^-4 5*10.^-4 10.^-3 10.^-2] ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sigma_list = [ 0.001 0.005 0.01 0.02 0.03 0.05]; % Variance  small to large
%sigma_list = [ 0.001  0.01 0.05 0.1]; % Variance  small to large
%sigma_list = [ 0.005 0.01 0.02 0.1 ];
sigma_list  = [ 0.01];
%sigma_list = [ 0.005 0.01 0.02 0.03  ];
%sigma_list = [ 0.005 0.007 0.01 0.015 ];
%sigma_list = [ 0.02 0.04 0.06 0.1  ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mu = 0.0001; K0=5.0; sigma= 0.001 0.01 0.1 
%%%
seed = 65;
%%seed = 65;
SimMax = 1;
t_period = 72000;
n_pull = 50;
T = t_period * n_pull;

n_Eimmigrant = [];
for Mu_itr= 1:size(MU_list,2)
    n_Eimmigrant = [n_Eimmigrant MU_list(Mu_itr)*T];
end
%disp(n_Eimmigrant)

n_total = [];
n_mean_del_T_list = [];
n_std_del_T_list= [];
%%% Stats Across Cells per Interval 
avgOnMeanNum = zeros( size(MU_list,2), size(K0_list,2), size(W_list,2), size(sigma_list,2), SimMax );
avgOnStdNum = zeros( size(MU_list,2), size(K0_list,2), size(W_list,2), size(sigma_list,2), SimMax );
avgOnStdNumTimeSpan = zeros( size(MU_list,2), size(K0_list,2), size(W_list,2), size(sigma_list,2), SimMax );
AvgNum = zeros( size(MU_list,2), size(K0_list,2), size(W_list,2), size(sigma_list,2), SimMax );

for Mu_itr= 1:size(MU_list,2)
    for K0_itr = 1:size(K0_list,2)
        for W_itr = 1:size(W_list,2)
            for Sig_itr = 1:size(sigma_list,2)
                n_event = [];
                mean_del_T_list = [];
                std_del_T_list = [];
                for SimTime = 1:SimMax
                    rng(seed+SimTime)
                    addpath('./.')
                    disp([ Mu_itr/size(MU_list,2) K0_itr/size(K0_list,2) W_itr/size(W_list,2) Sig_itr/size(sigma_list,2) SimTime/SimMax])
                    
                    
                    % intensity for immigrant Num immigrant
                    mu = MU_list(Mu_itr);
                  
                    % intensity for offspring Num offspring
                    K0 = K0_list(K0_itr);
                    
                    % time period for offspring

                    w=W_list(W_itr);
                    
                    
                    sigma=sigma_list(Sig_itr);
                    
                    [times_sim x_sim y_sim  offspring_sim clusterId_sim]= Hawkes_Simulation( mu, K0, w, T, sigma );
                    %disp( ['immigrant: ', num2str(sum(offspring_sim==0))] )
                    %%%%%%%%%%%%%%%%%%%%
                    [~, edges, xId] = histcounts( x_sim, (0:0.1:1) );
                    [~, edges, yId] = histcounts( y_sim, (0:0.1:1) );
                    xId = uint8(xId)-1;
                    yId = uint8(yId)-1;
                    Arm = yId*10 +xId;
                    [bin, edges, ~] = histcounts( Arm+0.01, (0:1:100) );
                    bin_re = reshape(bin, 10, 10).';
                    bin_re = bin_re(end:-1:1,:);
                    
                    t_edge = (0:t_period:T);
                    

                    %Counter Cluster
                    val = unique(clusterId_sim);
                    cnt = histc(clusterId_sim, val);
                    cnt_val = [cnt, val];
                    [B, I ] = sort(cnt);
                    keep_val = val( I(end:-1:end-(Top-1)) );


                    %%%%
                    cluster_keep = [];
                    disp(length(clusterId_sim))
                    for cluster_itr = 1 : length(clusterId_sim)
                        if length( find( keep_val == clusterId_sim(cluster_itr) ) ) > 0
                            cluster_keep = [cluster_keep; 1 ];
                        else
                            cluster_keep = [cluster_keep; 0 ];
                        end
                    end
                    
                    Period = t_period;
                    PullNums = ceil(Period*n_pull/Period);
                    NumArms = 100;
                    TimeSeperate = linspace( 0, Period*PullNums, PullNums+1);
                    TimeId = discretize(times_sim,TimeSeperate,'IncludedEdge','left');
                    
                    figure(2*(Mu_itr*1000+K0_itr*100+W_itr*10+ Sig_itr))
                    select = find(cluster_keep);
                    %scatter3(   x_sim( select ),  y_sim( select ),  times_sim( select ), clusterId_sim(select), '.');
                    
                    c_sim = clusterId_sim(select);
                    uniq = unique( c_sim );
                    temp = zeros(size(c_sim));
                    for t = 1:Top
                        temp(find(c_sim == uniq(t))) = t;
                    end
                    scatter3(  x_sim( select ),  y_sim( select ),  times_sim( select ),[] , clusterId_sim(select), '.');
                    top_out = [x_sim( select ),  y_sim( select ), times_sim( select ), temp];
                    
                    dlmwrite(['top_out.txt'], top_out)
                    %for arm = 1:NumArms
                            %Nums(arm, time) = numel( find(  (TimeId == time) & (Arm == arm) ) );
                            %time_plot = times_sim( find( Arm == arm) );
                            %xId_plot = xId( find( Arm == arm) );
                            %yId_plot = yId( find( Arm == arm) );
                            %cluster_plot = clusterId_sim( find( Arm == arm) );
                            %scatter3(   x_sim,   y_sim, times_sim, clusterId_sim,'.')
                     %       hold on 
                    %end
                    top_outset(gca,'Zticklabel',[])
                    set(gcf, 'Position', [300 50 600 500]);
                    set(gca,'ztick',linspace(0,t_period*n_pull,n_pull) )
                    set(gca,'xtick', (0:0.1:1))
                    set(gca,'ytick', (0:0.1:1))
                    %set(gca,'xtick',linspace(0,100,1) )
                    xlim([0 1])
                    ylim([0 1])
                    zlim([0 t_period*n_pull])
                    %colorbar
                    grid on 
                    
                    %{
                    figure(1*(SimTime*10000+Mu_itr*1000+K0_itr*100+W_itr*10+ Sig_itr))
                    
                    Nums = zeros(NumArms, PullNums);
                    %CellSeries = cell(NumArms, PullNums);
                    
                    for arm = 1:NumArms
                           % Nums(arm, time) = numel( find(  (TimeId == time) & (Arm == arm) ) );
                            time_plot = times_sim( find( Arm == arm) );
                            cluster_plot = clusterId_sim( find( Arm == arm) );
                            scatter(   repmat(  arm,  size(time_plot ,1) ,1), time_plot,256, cluster_plot,'.')
                            hold on 
                    end
        
                    set(gca,'Yticklabel',[])
                    set(gcf, 'Position', [300+SimTime*100 400 600 500]);
                    set(gca,'ytick',linspace(0,t_period*n_pull,n_pull) )
                    %set(gca,'xtick',linspace(0,100,1) )
                    ylim([0 t_period*n_pull])
                    xlim([0 100])
                    grid on 
                    %}
                    %{
                    figure(2*(Mu_itr*1000+K0_itr*100+W_itr*10+ Sig_itr))
                    for arm = 1:NumArms
                           % Nums(arm, time) = numel( find(  (TimeId == time) & (Arm == arm) ) );
                            time_plot = times_sim( find( Arm == arm) );
                            xId_plot = xId( find( Arm == arm) );
                            yId_plot = yId( find( Arm == arm) );
                            cluster_plot = clusterId_sim( find( Arm == arm) );
                            scatter3(   xId_plot,   yId_plot, time_plot, cluster_plot,'.')
                            hold on 
                    end
                    set(gca,'Zticklabel',[])
                    set(gcf, 'Position', [300 50 600 500]);
                    set(gca,'ztick',linspace(0,18000*200,200) )
                    set(gca,'xtick', (1:10))
                    set(gca,'ytick', (1:10))
                    %set(gca,'xtick',linspace(0,100,1) )
                    xlim([0 10])
                    ylim([0 10])
                    zlim([0 18000*200])
                    grid on 
                    %}
                    %figure(W_itr)
                    %hist(times_sim,200)
              
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %disp( ['Number of immigrant: '  num2str(length(find(offspring_sim == 0)))  ', Number of all data events: '  num2str(length(times_sim))])
                    % save out
                    % StrOut = ['./Matfiles/Spatial_Distribution_mu_' num2str(mu,'%1.6f') '_k0_' num2str(K0,'%1.6f') ...
                    %     '_omega_' num2str(w,'%1.6f')  '_sigma_' num2str(sigma,'%1.6f') '_T_' num2str(T,'%d') '_SimPath_' num2str(SimTime,'%03d') '.txt'];
                    %disp(StrOut)
                    %fileID = fopen( StrOut, 'w+');
                    %for d_pt = 1 : size(times_sim, 1)
                    %    fprintf(fileID,'%1.8f\t%1.8f\t%10d\t%d\n', y_sim(d_pt), x_sim(d_pt),  int64(times_sim(d_pt)), Arm(d_pt));
                    %end
                    %fclose( fileID );
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    
                    %%% Try to get time cluster %%%
                    del_T = [];
                    for c_id = min(val):1:max(val)
                        Idx = find(clusterId_sim == c_id);
                        delta_time = max(TimeId(Idx)) - min(TimeId(Idx));
                        del_T = [del_T delta_time];
                    end
                    mean_del_T = mean(del_T);
                    std_del_T = std(del_T);
                    %%%
                    mean_del_T_list = [mean_del_T_list mean_del_T];
                    std_del_T_list = [std_del_T_list std_del_T];
                    n_event = [n_event size(times_sim,1) ];
                end          
            end %Sig_itr
        end %W_itr 
         n_mean_del_T_list = [ n_mean_del_T_list; mean_del_T_list];
         n_std_del_T_list = [ n_std_del_T_list; std_del_T_list];
         n_total = [n_total ; n_event];
    end %K0_itr
end %Mu_itr
