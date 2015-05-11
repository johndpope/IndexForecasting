clc; clear all;
%% Read Data
data = csvread('data/returns.csv',1,1);

fid      = fopen('test.txt');
paramIds = textscan(fid,'%s',1,'HeaderLines',5);
fclose(fid);

%% Settings
ftse = 1; gsci = 2; bonds = 3; stocks = 4;

days         = size(data,1);
trading_days = 252;

%% Plot index cumulative returns
% PlotStandard(data);

%% Get daily returns + stats
daily_ret   = data(2:end,:) ./ data(1:end-1,:)-1;
overall_ret = geomean(daily_ret+1).^trading_days - 1; 

std_ret     = std(daily_ret)  * sqrt(trading_days);
cov_ret     = cov(daily_ret)  * trading_days;

%% Function to optimise
PortfolioReturns = @(w) (-w*overall_ret');

%% Optimisation for E-F
frontier_risks = .03:.005:.30;
frontier_len   = length(frontier_risks);
starting_w     = [0.7 0.3 .4 .6];
EF             = zeros(frontier_len,2);
weights        = zeros(frontier_len,4);
options        = optimset('MaxFunEvals',1000000, 'MaxIter', 1000000, ...
    'TolX', 0.000001);

% Loop through for frontier
for i = 1:frontier_len
    % starting_w = FindGridWeight(frontier_risks(i));
    [x, fval] = ... 
    fmincon(PortfolioReturns,starting_w,[],[],[],[],[],[],...
    @(w)(confun(w, cov_ret, frontier_risks(i))),optimoptions('fmincon'));
    EF(i,1)      = -fval;
    EF(i,2)      = sqrt(x*cov_ret*x');
    weights(i,:) = x;
end

%% Plot Results

inflection = find(diff(EF(:,1))<0,1);

plot(EF(1:inflection,2),EF(1:inflection,1),'b', ...
    EF(inflection:size(EF,1) ,2),EF(inflection:size(EF,1),1),'g--o');
xlabel('Portfolio Risk (Annual Vol)')
ylabel('Portfolio Returns')
title('Mean-Variance Efficient-Frontier')
grid on;


%% MaxDD & Ulcer Index for each point
% Function to calc DD
EF = EF(1:inflection,:);
dds = arrayfun (@(x) ((DrawDown(weights(x,:), daily_ret))),...
    1:inflection,'UniformOutput',false);
dds = cell2mat (dds')';

% Get max DD and Ulcer Index
maxDD       = max(dds);
ulcer_index = (sqrt(sum((dds*100).^2)/size(dds,1)));    % From Wiki

%% Fit frontier point and max drawdown
figure 

subplot(1,2,1) 
% VS vol
lin_fit_dd_vol = polyfit(EF(:,2)',maxDD,1);
% Plot results and fitted line
plot(EF(:,2),maxDD,EF(:,2),lin_fit_dd_vol(1)*EF(:,2)+lin_fit_dd_vol(2));
xlabel('Portfolio Risk (Annual Vol)')
ylabel('Max DrawDown')
title('MaxDD vs. Vol')

% VS return
subplot(1,2,2)
lin_fit_dd_ret = polyfit(EF(:,1)',maxDD,1);
% Plot results and fitted line
plot(EF(:,1),maxDD,EF(:,1),lin_fit_dd_ret(1)*EF(:,1)+lin_fit_dd_ret(2));
xlabel('Portfolio Returns')
ylabel('Max DrawDown')
title('MaxDD vs. Returns')

%% Fit frontier point and Ulcer Index
figure 
subplot(1,2,1)

% VS vol
lin_fit_dd_vol = polyfit(EF(:,2)',ulcer_index,1);
% Plot results and fitted line
plot(EF(:,2),ulcer_index,EF(:,2),lin_fit_dd_vol(1)*EF(:,2)+lin_fit_dd_vol(2));
xlabel('Portfolio Risk (Annual Vol)')
ylabel('Ulcer Index')
title('Ulcer vs. Vol')

% VS return
subplot(1,2,2)
lin_fit_dd_ret = polyfit(EF(:,1)',ulcer_index,1);
% Plot results and fitted line
plot(EF(:,1),ulcer_index,EF(:,1),lin_fit_dd_ret(1)*EF(:,1)+lin_fit_dd_ret(2));
xlabel('Portfolio Returns')
ylabel('Ulcer Index')
title('Ulcer vs. Returns')
