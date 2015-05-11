function [ port_rets ] = MomSearch(all_ret, day_start, day_end, ... 
    vol_target, dd_target, lookback, holding)
% Function to find best look back period for portfolio
port_rets      = 0;
rf             = .03;
ftse           = 1;
gsci           = 2;
bonds          = 3;
sectors        = 4:13;
sp500          = 14;
trading_days   = 252;

start          = day_start-1;
total_days     = day_end-day_start+1;

sect_ret       = all_ret(:,sectors);
periods        = ceil(total_days / holding); % total holding periods 

mom_port      =  @(offset)...
    (momentum(sect_ret,start,offset,holding,lookback,total_days)); 

mom_rets       = arrayfun(@(x) (mom_port(x)), 1:periods,'UniformOutput',false);
momentum_port  = cell2mat(mom_rets');


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
starting_w     = [ 0.5 1 0.2 0.5 0.5 ];
for i = 1:5
    op = optimset('MaxFunEvals',10000000, 'MaxIter', 10000000, ...
        'TolX', 0.00000001,'Display', 'off');

    [weights, value] = fmincon(PortfolioReturns,starting_w,[],[],[],[],[],[],...
        @(w)(confun2(w, cov_ret, vol_target, universe_daily_ret, dd_target)),op);

    port_rets       = max(port_rets,-PortfolioReturns(weights));
    starting_w      = weights;
end 
end

