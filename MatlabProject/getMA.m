function [ MA ] = getMA(price, lag)
% Get the MA over entire period with given lag
total_days   = size(price,1);
MA = [];

for i = lag:total_days
start = i - lag + 1;
current_period_MA = mean(price(start:i,:));

MA = [MA ; current_period_MA];
end

end