function [mom_values] = getMomInd(price, lag)
% Get the Momentum Indicator over entire period with given lag
total_days   = size(price,1);
mom_values = [];

for i = (lag+1):total_days
start = i - lag;
current_period_mom = price(i,:) - price(start,:);

mom_values = [mom_values ; current_period_mom];
end

end

