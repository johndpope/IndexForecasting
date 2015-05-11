function [ corrMat ] = getCorrMatrix(returns, lag, target_index)
% Get the correlation over entire period with given lag
total_days = size(returns,1);
corrMat    = [];

for i = lag:total_days
start = i - lag + 1;
current_period_corr = corr(returns(start:i,target_index),returns(start:i,:));

corrMat = [corrMat ; current_period_corr];

end
end