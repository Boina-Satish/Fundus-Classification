function [Accu, f1Score] = SVMclassi(DTrain, Y)

X = DTrain;

%%
classOrder = unique(Y);
rng(1); 
t = templateSVM('Standardize',true,'KernelFunction','gaussian');
PMdl = fitcecoc(X,Y,'Learners',t,'ClassNames',classOrder)
                   %returns a full, trained, multiclass, error-correcting 
                   %output codes (ECOC) model using the predictors in table X 

% Mdl = PMdl.Trained{1};      %Extract trained,compact classifier
% testInds = test(PMdl.Partition);    %Extract the test indices
% XTest = X(testInds,:);
% YTest = Y(testInds,:);

labels = predict(PMdl,X);
[C,order] = confusionmat(Y,labels) %gives the confusion matrix
                                  
TP1 = C(1,1);  %True Positive for Class 1
FN1 = C(1,2)+C(1,3); %False Negative for Class 1
FP1 = C(2,1) + C(3,1); %False Positive for Class 1
TN1 = C(2,2) + C(2,3) + C(3,2) + C(3,3); %True Negative for Class 1 

Accu = (C(1,1)+C(2,2)+C(3,3))/sum(sum(C)) %Accuracy 
Prec1 = TP1/(TP1+FP1); %Precision
Sens1 = TP1/(TP1+FN1); %Sensitivity or Recall
Spec1 = TN1/(TN1+FP1); %Specificity

TP2 = C(2,2);
FN2 = C(2,1)+C(2,3);
FP2 = C(1,2) + C(3,2);
TN2 = C(1,1) + C(1,3) + C(3,1) + C(3,3);

Prec2 = TP2/(TP2+FP2);
Sens2 = TP2/(TP2+FN2);
Spec2 = TN2/(TN2+FP2);

TP3 = C(3,3);
FN3 = C(3,1)+C(3,2);
FP3 = C(1,3) + C(2,3);
TN3 = C(1,1) + C(1,2) + C(2,1) + C(2,2);

Prec3 = TP3/(TP3+FP3);
Sens3 = TP3/(TP3+FN3);
Spec3 = TN3/(TN3+FP3);

AvgPrec = (Prec1+Prec2+Prec3)/3; %Average of Precision
AvgSens = (Sens1+Sens2+Sens3)/3; %Average of Sensitivity

f1Score = 2*((AvgPrec*AvgSens)/(AvgPrec+AvgSens)) %F1Score
