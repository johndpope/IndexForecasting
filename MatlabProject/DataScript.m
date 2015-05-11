clc; clear all;
%% Read Data
var_filename = 'project_vars.mat';

SUB_FILE  = 'data/sub_returns.csv';
sub_data  = csvread(SUB_FILE,1,1);

fid       = fopen(SUB_FILE);
subIds    = textscan(fid,'%s',size(sub_data,2)+1,'HeaderLines',0,'Delimiter',',');
formatSt  = ['%s' repmat('%*s',1,size(sub_data,2))];
dates     = textscan(fid,formatSt,'Delimiter',',');
dates     = datenum(dates{1});
fclose(fid);

%% Settings
PortfolioReturns = @(w) (-w*overall_ret'); % Portfolio returns function

indicies     = subIds{1}(2:end);
sect_dic     = getSectors();
total_days   = size(sub_data,1);
trading_days = 252;
% prices       = mat2dataset(sub_data,'VarNames',indices);

%plot(dates,prices)
%datetick('x','yyyy','keeplimits')

%% GET FEATURES OF DATA

% Get daily returns + volatility stats
daily_ret         = sub_data(2:end,:) ./ sub_data(1:end-1,:)-1;
overall_ret       = geomean(daily_ret+1).^trading_days - 1; 
vol_10            = getVol(daily_ret, 10);
vol_20            = getVol(daily_ret, 20);

% Get the 1/4/13/52 cumulative week returns:
cum_ret_1   = getCumRet(daily_ret, 5*1);
cum_ret_4   = getCumRet(daily_ret, 5*4);
cum_ret_13  = getCumRet(daily_ret, 5*13);
cum_ret_52  = getCumRet(daily_ret, 5*52);

% Get 20 day and 50 day Simple Moving Averages
MA_20       = getMA(sub_data, 20);
MA_50       = getMA(sub_data, 50);
EMA_20      = getEMA(sub_data, 20);
EMA_50      = getEMA(sub_data, 50);

% GET CORRELATION
corr_mat_sp500 = getCorrMatrix(daily_ret, 20, 1);

% Get the Correlation Surprise index
surprises_ind = getCorrSurprise(daily_ret, 3);

% Get Momentum Indicator 10 day
mom_ind = getMomInd(sub_data, 10);

%% SAVE WORKSPACE DATA
clear ans fid formatSt SUB_FILE subIds;
save(var_filename);
disp('DONE');