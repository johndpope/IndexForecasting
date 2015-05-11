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

CUTOFF = 300;

TRAIN_X = X(1:end-CUTOFF,:);
TRAIN_Y = Y(1:end-CUTOFF);

TEST_X  = X(end-CUTOFF+1:end,:);
TEST_Y  = Y(end-CUTOFF+1:end);

ind = SPX_IND;
%% Multinomial logistic regression
% Fit classifier:
[mlr_clf,dev,stats] = mnrfit(TRAIN_X, TRAIN_Y+POS_ADJ);

% LL = stats.beta - 1.96.*stats.se;
% UL = stats.beta + 1.96.*stats.se;

% Use fitted classifier for prediction:
pred_Y_prob = mnrval(mlr_clf,TEST_X);

% Get the highest probability 
[Y_proba, Y_Ind] = max(pred_Y_prob,[],2);
pred_Y = Y_Ind - POS_ADJ;
%% Find all where we buy/sell (i.e. predict larger return than threshold)
I = find(abs(pred_Y)>0);
[pred_Y(I) TEST_Y(I)]
prob_correct = sum(TEST_Y(I)-pred_Y(I) == 0) / length(I)

 
STRAT_Performance = prod((daily_ret(end-CUTOFF+1:end,ind)).*pred_Y + 1)
SPX_Performance   = prod((daily_ret(end-CUTOFF+1:end,ind))+1)


%% 
% options.MaxIter = 100000;
% svm_struct = svmtrain(X(1:end-100,:), Y(1:end-100), 'Options', options); 
% %%
% pred_Y = svmclassify(svm_struct,X((end-99):end,:));
% % [Y((end-99):end,:) Group]
% sum(abs(Y((end-99):end,:)-Group)) / length(Group)
% % SVMModel = fitcsvm(X,classes,'KernelFunction','rbf','Standardize',true,'ClassNames',{'negClass','posClass'});


%% K Fold - SVM
tic
options.MaxIter = 100000;
probs = zeros(10,1);
for i = 1:10
    i
    training_X = X;
    training_Y = Y;
    
    ind = ((i-1)*300+1) : (i*300);
    testing_X = X(ind,:);
    testing_Y = Y(ind);
    training_X(ind,:)=[];
    training_Y(ind,:)=[];
    
    svm_struct = svmtrain(training_X, training_Y, 'Options', options);
    pred_Y = svmclassify(svm_struct,testing_X);
    [pred_Y testing_Y]
    probs(i) = 1 - sum(abs(testing_Y-pred_Y)) / length(pred_Y);
    probs(i)
end
disp('SVM K-FOLD RESULTS');
disp(probs)
mean(probs)
toc
%% K Fold - Neural Nets




