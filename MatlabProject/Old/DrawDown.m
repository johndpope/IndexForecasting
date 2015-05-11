function [ drawdowns ] = DrawDown( w, daily_ret )
% Get cumulative returns
port_ret = w * daily_ret' + 1;
cum_ret  = cumprod(port_ret);

% Calculate difference with the max point
dd = @(index) ((max(cum_ret(1:index)) - cum_ret(index))...
    /max(cum_ret(1:index))); 

drawdowns = arrayfun (dd, 1:length(cum_ret));
end

