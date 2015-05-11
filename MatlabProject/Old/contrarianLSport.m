clc; clear all;

%% Read Data
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

% Calculated from Excel
best_sharpe    = [1 4 7 8] + 3; % From Excel
weights        = ones(4,1)/4;

daily_ret   = data(day_start:day_end,:) ./ data(day_start-1:day_end-1,:)-1;
% overall_ret = (data(day_end,:) ./ data(day_start,:)).^(1/12.5) - 1
overall_ret = geomean(daily_ret+1).^trading_days - 1;
%% Momentum 
lookback       = 120;
holding_period = trading_days/12;

% lookback = 90;
%holding_period = 80;

start          = day_start-1;
total_days     = day_end-day_start+1;
all_ret        = data(2:days,:) ./ data(1:days-1,:)-1;
sect_ret       = all_ret(:,sectors);
periods        = ceil(total_days / holding_period); % total holding periods 

%% Find Momentum Strat Returns
mom_port      =  @(offset)...
    (contrarian_LS(sect_ret,start,offset,holding_period,lookback,total_days)); 

mom_rets       = arrayfun(@(x) (mom_port(x)), 1:periods,'UniformOutput',false);
momentum_port  = cell2mat(mom_rets');
mom_annual_ret = (geomean(momentum_port+1)^trading_days) -1
mom_std        = std(momentum_port) * sqrt(trading_days)
mom_SR         = (mom_annual_ret-rf)/mom_std
mom_DD         = DrawDown(1, momentum_port);
mom_maxDD      = max(mom_DD)
mom_ulcer_ind  = (sqrt(sum((mom_DD*100).^2)/length(mom_DD)))

figure
plot(cumprod(momentum_port+1))
title('Momentum Strat (LB:90  H:80) Cum-Returns')
xlabel('Time (days)')
ylabel('Standardised Returns')

%% Optimise Portfolio - Setup
% Vol 14%, dd 12%, no borrowing  GSCI + FTSE overlay
% sp500 + bonds + mom = 100%
vol_target         = 0.18;
dd_target          = 0.18;
non_sector_indices = [sp500, bonds, gsci, ftse];

non_sector         = all_ret(start:(start+total_days-1),non_sector_indices);

% All indicies: to optimise over - sp500, bonds, gsci, ftse, momentum
universe_daily_ret = [non_sector, momentum_port];

% Get geomean returns and covar matrix
overall_ret = geomean(universe_daily_ret+1).^trading_days - 1;
cov_ret     = cov(universe_daily_ret)  * trading_days;
% Optimisation function
PortfolioReturns = @(w) (-w*overall_ret' + (w(1)+w(2)+w(5)-1)*rf);

%% Run optimisation
% starting_w     = [-0.5496 0.9837 -0.4749 0.6120 0.5672];
% starting_w     = [-0.2969 1.6300 -0.4568 1.0049 0.6495];
% starting_w     = [ 0.2223    0.4804   -0.6018    1.0370    0.2984];
% starting_w     = [ -0.4111    1.6394   -0.3508    1.0190    0.7567 ];
starting_w = [ 0.5 1 0.2 0.5 0.5 ];  
op = optimset('MaxFunEvals',10000000, 'MaxIter', 10000000, ...
    'TolX', 0.00000001, 'Display', 'off');

for i = 1:5
[weights, value] = fmincon(PortfolioReturns,starting_w,[],[],[],[],[],[],...
    @(w)(confun2(w, cov_ret, vol_target, universe_daily_ret, dd_target)),op)

BestPort_Ret    = -PortfolioReturns(weights)
BestPort_STD    = sqrt(weights*cov_ret*weights')
BestPort_Sharpe = (BestPort_Ret-rf)/BestPort_STD
dd              = DrawDown(weights, universe_daily_ret);
maxDD           = max(dd)
ulcer_ind       = (sqrt(sum((dd*100).^2)/length(dd)))
% [a b] = confun2(weights, cov_ret, vol_target, universe_daily_ret, dd_target)
portfolio = universe_daily_ret*weights';
starting_w = weights;
end 
plot(cumprod(portfolio+1))
title('Optimal Portfolio Cum-Returns (w/ Mom LB:90  H:80)')
xlabel('Time (days)')
ylabel('Standardised Returns')

%% Find best look back and holding period for best portfolio
holding_days  = 20:10:120;
lookback_days = 50:10:120;

port_returns  = zeros(length(holding_days), length(lookback_days));

for i = 1:length(holding_days)
    for j = 1:length(lookback_days)
        [ i j ]
        port_returns(i,j) = MomSearch(all_ret, day_start, day_end, ... 
         vol_target, dd_target, lookback_days(j), holding_days(i));
        port_returns(i,j)
    end
end

port_returns
%% Best return
% r=7;c=5;   
% holding_days(r)
% lookback_days(c)



