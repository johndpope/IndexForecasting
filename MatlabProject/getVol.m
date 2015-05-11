function [ vol ] = getVol(returns, lag)
% Get the volatilities over entire period with given lag
trading_days = 252;
total_days   = size(returns,1);
vol = [];

for i = lag:total_days
start = i - lag + 1;
current_period_vol = std(returns(start:i,:))*sqrt(trading_days);

vol = [vol ; current_period_vol];
end

end