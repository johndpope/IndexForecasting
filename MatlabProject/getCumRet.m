function [ cum_ret ] = getCumRet(returns, lag)
% Get the cumulative returns over the given period
total_days   = size(returns,1);
cum_ret = [];

for i = lag:total_days
start = i - lag + 1;
current_period_cum_ret = prod(returns(start:i,:)+1)-1;

cum_ret = [cum_ret ; current_period_cum_ret];
end
end