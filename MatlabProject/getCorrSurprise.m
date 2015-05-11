function [surprises] = getCorrSurprise( returns, look_back)

trading_days = 252;
lag = look_back * trading_days;

surprises    = [];
total_days   = size(returns,1);


for i = lag:total_days
start     = i - lag + 1;

mean_rets = mean(returns(start:(i-1),:));
cov_rets  = cov(returns(start:(i-1),:));% *trading_days;

[ dt mag_sup corr_sup ] = getTurbulance( mean_rets, returns(i,:), cov_rets);

surprises = [surprises ; dt mag_sup corr_sup];
end

end


