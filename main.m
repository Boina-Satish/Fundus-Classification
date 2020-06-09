clc;
clear all;
close all;
cd database %All the dataset are stored in "database" folder
DTrain = []; %To save the featuress
 
for i = 1:671
    st=int2str(i);
    str=strcat(st,'.tif')
    rgbimage=imresize(imread(str),[1500,1500]);
%%  
    FL =[0 1 0 1 0 1 0 1 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 1 ...
        0 1 0 1 0 1 1 1 1 1 0 1 0 1 0 0 0 1 1 0 1 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 1 1 1 0 1 0 1 0 ...
        1 1 0 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0 ...
        1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 0 0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 ...
        1 1 0 1 0 1 1 1 1 1 1 0 1 0 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 0 1 1 ...
        1 0 1 0 0 0 0 1 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 0 0 1 1 0 0 0 1 0 0 0 0 1 0 1 1 1 1 0 1 ...
        1 0 1 1 0 1 0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 0 1 1 0 0 0 1 0 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 ...
        1 0 1 1 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 1 0 1 0 0 1 1 0 1 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 1 0 1 0 0 ...
        1 0 1 1 0 1 0 0 1 1 0 1 1 0 1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 ...
        1 0 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 1 0 1 0 1 0 0 1 0 1 1 0 0 1 1 0 1 0 0 0 1 1 0 ...
        1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1 0 1 0 0 1 1 0 0 1 1 0 0 0 ...
        0 1 1 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0 0 1 0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 0 0 0 ...
        1 1 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 0 1 0 1 0 0 0 1 1 0 0 1 0 1 0 1 0 ...
        1 1 1 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 ];
          %FL is the matrix storing 1's and 0's to bring the Optic Disk to
          %the right side   
%% Flipping the images horizontally
    if FL(i) == 1   %for flipping the image to get the OD in the right side of the image
        rgbimage = flip(rgbimage,2);
%         figure, imshow(imR);
    else
        rgbimage = rgbimage;
    end
    
    redChannel = rgbimage(:, :, 1);
    greenChannel = rgbimage(:, :, 2);
    
%% Applying Median Filter
    redMedian = medfilt2(redChannel);
    greenMedian = medfilt2(greenChannel);

%% Applying LBP Function
    y= zeros(size(rgbimage));
    LBPGimage = LBP_fn(greenMedian);
    
%% Finding OD and Vessel Masks
%     od_mask = od_seg(rgbimage);
    od_mask = od_seg_1(rgbimage,i); %OD mask is obtained
%     figure, imshow(od_mask);
    v_mask = vessel_seg(rgbimage); %Vessel mask is obtained
%      figure, imshow(v_mask);

%% Segmentation
    im2 = double(LBPGimage);
    im5 = double(greenMedian);
    
%% Optic disk segmentation
    im_od = double(od_mask);
    LBP_GsegOd = im_od.*im2; %OD segmentation on LBP image
    Orig_GsegOd = im_od.*im5; %OD segmentation on green Channel of the original image
    
%% Vessel Segmentation
    im_v = double(v_mask);
    LBP_GsegV = im_v.*LBP_GsegOd; %Vessel segmentation on LBP image
    Orig_GsegV = im_v.*Orig_GsegOd; %Vessel segmentation on green Channel of the original image
  
%% glcm
    [glcms,SI] = graycomatrix(Orig_GsegV); %To find Gray-level Co-occurrence Matrix
    stats = graycoprops(glcms); %To find properties of Gray-level Co-occurrence Matrix
    en=stats.Energy;
    corre=stats.Correlation;
    home=stats.Homogeneity;
    cont=stats.Contrast;
   
%% Features
    Gmean = mean2(Orig_GsegV);
    Gstd = std2(Orig_GsegV);
    Gmedian = median(Orig_GsegV(:));
    Gentropy = entropy(Orig_GsegV);
    Gskew = skewness(Orig_GsegV(:));
    Gkurt = kurtosis(Orig_GsegV(:));
    [m,n] = size(LBP_GsegV);
    featureall = [corre en home cont Gmean Gstd Gmedian Gentropy Gskew Gkurt sum(sum(LBP_GsegV))/(m*n)];
    DTrain = [DTrain;featureall]; %All the features of the dataset stored here
    
end

cd ..

%% Feature Selection
load DTrain
X = DTrain;
%%

Y = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;...
    3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;...
    1;1;1;1;1;1;1;1;1;1;1;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;...
    3;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;...
    2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;3;3;3;3;3;3;3;3;3;3;3;...
    3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;...
    3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;...
    3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;3;...
    3;3;3;3;3;3;3;3];
         %Y is a matrix having 1's,2's and 3's
         %Size of Y is equal to total number of datasets
         % 1 - AMD image,  2 - Normal\ Healthy image,  3 - DR image
%% Naive Bayes Classification
classOrder = unique(Y);
rng(1); 
K = numel(unique(Y));
options = statset('UseParallel',true);
t = templateNaiveBayes('DistributionNames','mvmn')
PMdl = fitcecoc(X,Y,'Learners',t,'ClassNames',classOrder,'Options',options,'Coding','onevsone');
                   %returns a full, trained, multiclass, error-correcting 
                   %output codes (ECOC) model using the predictors in table X 

% Mdl = PMdl.Trained{1};      %Extract trained,compact classifier
% testInds = test(PMdl.Partition);    %Extract the test indices
% XTest = X(testInds,:);
% YTest = Y(testInds,:);

labels = predict(PMdl,X);

% idx = randsample(sum(testInds),20); % Shows 20 outputs randomly
% table(YTest(idx),labels(idx),'VariableNames',{'TrueLabels','PredictedLabels'})
%% 
l = loss(PMdl,X,Y) %Calculates the loss factor
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

%% Support Vector Machine (SVM) classification
[Accu_SVM, f1Score_SVM] = SVMclassi(DTrain,Y)




