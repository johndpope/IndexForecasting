function [ returns ] = ...
    contrarian( sect_daily_ret, start, offset, holding_period,lookback,total_days)
% Find best performing sectors in past period
NO_OF_BOTTOM_SECTORS = 4;

% Setup time periods
CURRENT_TIME    = start + (offset-1) * holding_period;
LOOKBACK_PERIOD = (CURRENT_TIME-lookback):(CURRENT_TIME-1);

% Check final period 
CURRENT_PERIOD_END = start+min(total_days,offset*holding_period)-1;
CURRENT_PERIOD  = CURRENT_TIME:CURRENT_PERIOD_END;

% Find overall geometric mean returns
overall = geomean(sect_daily_ret(LOOKBACK_PERIOD,:)+1).^252 -1;
% vol     = std(sect_daily_ret(LOOKBACK_PERIOD,:)) * sqrt(252);
% sharpe  = (overall - 0.03)./vol;

% Sort overall returns and take top 4
[ ~ ,sortIndex] = sort(overall,'ascend');
returns = sect_daily_ret(CURRENT_PERIOD,sortIndex(1:NO_OF_BOTTOM_SECTORS));
returns = mean(returns,2);
end

