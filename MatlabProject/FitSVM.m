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
%% Fit SVM 
options.MaxIter = 100000;
tic
disp('SVM: Beginning Training...')
% Fit classifier:
svm_struct = svmtrain(TRAIN_X, TRAIN_Y, 'Options', options);

toc
disp('Training Complete!')

% Use fitted classifier for prediction:
pred_Y = svmclassify(svm_struct,TEST_X);

%% Find all where we buy/sell (i.e. predict larger return than threshold)

[pred_Y TEST_Y]
prob_correct = sum(TEST_Y == pred_Y) / CUTOFF

disp(['Performance over ' num2str(CUTOFF) ' days:'])
STRAT_Performance = prod((daily_ret(end-CUTOFF+1:end,SPX_IND)).*pred_Y + 1)
SPX_Performance   = prod((daily_ret(end-CUTOFF+1:end,SPX_IND))+1)

