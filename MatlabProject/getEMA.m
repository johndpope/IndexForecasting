function [ EMA ] = getEMA(price, lag)
% Get the EMA over entire period with given lag
total_days   = size(price,1);

alpha = 2/(lag+1);
EMA = [];

for i = lag:total_days
start = i - lag + 1;

if i == lag
current_period_EMA = mean(price(start:i,:));
else 
current_period_EMA = EMA(start-1,:) + alpha * (price(i,:)-EMA(start-1,:));
end

EMA = [EMA ; current_period_EMA];
end

end