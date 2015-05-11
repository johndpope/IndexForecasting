function [ sharpe ] = ...
    mom_sharpe( sect_ret, trading_days, rf, start, periods, holding_period,... 
    total_days, lookback )
% Function to optimise - find best lookback period for momentum strat 
% Use Sharpe Ratio as comparison


mom_port      =  @(offset)...
    (momentum(sect_ret,start,offset,holding_period,lookback,total_days)); 

mom_rets      = arrayfun(@(x) (mom_port(x)), 1:periods,'UniformOutput',false);
momentum_port = cell2mat(mom_rets');


mom_annual_ret = (geomean(momentum_port+1)^trading_days) -1;
mom_std        = std(momentum_port) * sqrt(trading_days);
sharpe         = (mom_annual_ret-rf)/mom_std;

end

