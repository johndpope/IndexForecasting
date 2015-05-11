% clear all; clc;

% load fisheriris

% I = randperm(30);
% train_label={zeros(30,1),ones(30,1),2*ones(30,1)};
% train_cell={meas(1:30,:),meas(51:80,:),meas(101:130,:)};
% 
% 
% [svmstruct] = DSVM.Train_DSVM(train_cell,train_label);
% label=[0 1 2];
% 
% test_mat=[meas(31:40,:);meas(81:90,:);meas(131:140,:)];
% 
% [Class_test] = DSVM.Classify_DSVM(test_mat,label,svmstruct);
% 
% labels=[zeros(1,10),ones(1,10),2*ones(1,10)];
% 
% [Cmat,DA]= DSVM.confusion_matrix(Class_test,labels,{'A','B','C'})

I_1 = find(Y==-1);
I_2 = find(Y==0);
I_3 = find(Y==1);

I_1_cut = round(length(I_1)*0.9);
I_2_cut = round(length(I_2)*0.9);
I_3_cut = round(length(I_3)*0.9);

train_label={Y(I_1(1:I_1_cut)),Y(I_2(1:I_2_cut)),Y(I_3(1:I_3_cut))};

train_cell={X(I_1(1:I_1_cut),:),X(I_2(1:I_2_cut),:),X(I_3(1:I_3_cut),:)};

[svmstruct] = DSVM.Train_DSVM(train_cell,train_label);
%% 
label=[-1 0 1];

test_mat=[X(I_1(I_1_cut+1:end),:);X(I_2(I_2_cut+1:end),:);X(I_3(I_3_cut)+1:end,:)];

[Class_test] = DSVM.Classify_DSVM(test_mat,label,svmstruct);

labels=[Y(I_1(I_1_cut+1:end));Y(I_2(I_2_cut+1:end));Y(I_3(I_3_cut+1:end))];
%% 
[Cmat,DA]= DSVM.confusion_matrix(Class_test',labels,{'A','B','C'})

trainmsvm(X,Y,'-m WW -k 1', 'mymsvm');
