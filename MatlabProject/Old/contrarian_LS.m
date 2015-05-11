function [ returns ] = ...
    contrarian_LS( sect_daily_ret, start, offset, holding_period,lookback,total_days)
% Find best performing sectors in past period

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
best  = sect_daily_ret(CURRENT_PERIOD,sortIndex(1:4));
worst = sect_daily_ret(CURRENT_PERIOD,sortIndex(7:10));
returns = sum(best*0.5,2) - sum(worst*0.25, 2);
end

