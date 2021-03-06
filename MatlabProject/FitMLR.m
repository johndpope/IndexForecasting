clc; clear all;
%% Read All Data (run DataScript.m first)
% Call getTrainingData
var_filename = 'project_vars.mat';
load(var_filename);

POS_ADJ = 2;                           % A const to make positive class ints

SPX_IND = 1;
[X, Y, X_ALL]  = getTrainingData(SPX_IND, daily_ret, vol_10, vol_20, cum_ret_1,...
    cum_ret_4, cum_ret_13, cum_ret_52, MA_20, MA_50, EMA_20, EMA_50, mom_ind, ... 
    corr_mat_sp500, surprises_ind);      % 25 Variables

CUTOFF = 252;

TRAIN_X = X(1:end-CUTOFF,:);
TRAIN_Y = Y(1:end-CUTOFF);

TEST_X  = X(end-CUTOFF+1:end,:);
TEST_Y  = Y(end-CUTOFF+1:end);
%% Multinomial logistic regression
tic
disp('Logistic Reg: Beginning Training...')
% Fit classifier:

warning('off','MATLAB:nearlySingularMatrix')
warning('off','stats:mnrfit:IterOrEvalLimit')
[mlr_clf,dev,stats] = mnrfit(TRAIN_X, TRAIN_Y+POS_ADJ);

toc
disp('Training Complete!')
% LL = stats.beta - 1.96.*stats.se;
% UL = stats.beta + 1.96.*stats.se;

% Use fitted classifier for prediction:
pred_Y_prob = mnrval(mlr_clf,TEST_X);

% Get the highest probability 
[Y_proba, Y_Ind] = max(pred_Y_prob,[],2);
pred_Y = Y_Ind - POS_ADJ;
%% Find all where we buy/sell (i.e. predict larger than threshold)
threshold = 0.5;
I = find(Y_proba>=threshold);
% I = find(Y_Ind)
[pred_Y(I) TEST_Y(I)];
prob_correct = sum(TEST_Y(I)-pred_Y(I) == 0) / length(I)

disp(['Performance over ' num2str(CUTOFF) ' days:'])
STRAT_Performance = prod((daily_ret(end-CUTOFF+1:end,SPX_IND)).*((Y_proba>threshold) .* pred_Y) + 1)
SPX_Performance   = prod((daily_ret(end-CUTOFF+1:end,SPX_IND))+1)


%% Modified Kelly Criterion Sizing
sizing = (1./(1-Y_proba)-1) .* pred_Y;

disp(['Performance over ' num2str(CUTOFF) ' days:'])
STRAT_Performance_sized = prod((daily_ret(end-CUTOFF+1:end,SPX_IND)).*sizing + 1)
SPX_Performance_sized   = prod((daily_ret(end-CUTOFF+1:end,SPX_IND))+1)
