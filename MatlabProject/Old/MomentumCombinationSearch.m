clear all; clc;
data = csvread('../data/Part2.csv',1,1);

rf             = 0.03;

ftse           = 1;
gsci           = 2;
bonds          = 3;
sectors        = 4:13;
sp500          = 14;

days           = size(data,1);
day_start      = 249;
day_end        = days-1;
trading_days   = 252;
daily_ret   = data(day_start:day_end,:) ./ data(day_start-1:day_end-1,:)-1;

start          = day_start-1;
total_days     = day_end-day_start+1;
all_ret        = data(2:days,:) ./ data(1:days-1,:)-1;
sect_ret       = all_ret(:,sectors);

vol_target         = 0.18;
dd_target          = 0.18;

%% Find Momentum Strat Returns
lookbacks = [ 50 70 90];
holdings  = [ 40 70 90 120];
total     = length(lookbacks)*length(holdings);

momentums = zeros(3032,total);
count = 1;

for i = 1:length(lookbacks)
   for j = 1:length(holdings)
       periods        = ceil(total_days / holdings(j)); % total holding periods 
        mom_port      =  @(offset)...
            (momentum(sect_ret,start,offset,holdings(j),lookbacks(i),total_days)); 

        mom_rets       = arrayfun(@(x) (mom_port(x)), 1:periods,'UniformOutput',false);
        
        momentums(:,count)  = cell2mat(mom_rets');
        [count lookbacks(i) holdings(j)]
        count = count + 1;
    end    
end

%% Run optimisation
% Optimisation function
non_sector_indices = [sp500, bonds, gsci, ftse];

non_sector         = all_ret(start:(start+total_days-1),non_sector_indices);

% All combinations of 2 momentum portfolios
combinations_2     = combnk(1:12,2);
combinations_3     = combnk(1:12,3);
%% 
starting_w     = [ 0.5 1 0.2 0.5 0.25 0.25 0.1 ];
op = optimset('MaxFunEvals',10000000, 'MaxIter', 10000000, ...
        'TolX', 0.00000001,'Display', 'off');
    
best_port_3 = zeros(length(combinations_3),4);

for i = 1:length(combinations_3)
    i
    % momentums to select
    select = combinations_3(i,:);
    % 6 Assets
    universe_daily_ret = [non_sector, momentums(:,select)];
    overall_ret = geomean(universe_daily_ret+1).^trading_days - 1;
    cov_ret     = cov(universe_daily_ret)  * trading_days;
    
    % Redefine port returns calculation with new returns
    PortfolioReturns = @(w) (-w*overall_ret' + (w(1)+w(2)+w(5)+w(6)+w(7)-1)*rf);
    port_rets = 0;
    % Find best
    for j = 1:3
       [weights, value] = fmincon(PortfolioReturns,starting_w,[],[],[],[],[],[],...
        @(w)(confun_comb_mom(w, cov_ret, vol_target, universe_daily_ret, dd_target)),op);

        port_rets       = max(port_rets,-PortfolioReturns(weights));
        starting_w      = weights;
    end
    
    best_port_3(i,:) = [select port_rets];
    [select port_rets]
end

best_port_3

%% Get best

select = [1 3 9];
% 6 Assets
universe_daily_ret = [non_sector, momentums(:,select)];
overall_ret = geomean(universe_daily_ret+1).^trading_days - 1;
cov_ret     = cov(universe_daily_ret)  * trading_days;

% Redefine port returns calculation with new returns
PortfolioReturns = @(w) (-w*overall_ret' + (w(1)+w(2)+w(5)+w(6)+w(7)-1)*rf);
port_rets = 0;
% Find best
for j = 1:15
   [weights, value] = fmincon(PortfolioReturns,starting_w,[],[],[],[],[],[],...
    @(w)(confun_comb_mom(w, cov_ret, vol_target, universe_daily_ret, dd_target)),op);

    port_rets       = max(port_rets,-PortfolioReturns(weights));
    starting_w      = weights;
    
    port_rets
    starting_w
    
    BestPort_STD    = sqrt(weights*cov_ret*weights')
    BestPort_Sharpe = (port_rets-rf)/BestPort_STD
    dd              = DrawDown(weights, universe_daily_ret);
    maxDD           = max(dd)
    ulcer_ind       = (sqrt(sum((dd*100).^2)/length(dd)))
end
%% 

plot(cumprod(universe_daily_ret*weights'+1))
title('Optimal Portfolio Cum-Returns (w/ Mom1 LB:50  H:40 & Mom2 LB:70 H:40 & Mom3 LB:90 H:40)')
xlabel('Time (days)')
ylabel('Standardised Returns')
grid on