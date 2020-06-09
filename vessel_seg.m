function imF = vessel_seg(inputImage)

% inputImage = imread('image006.png');
% goundTruth = imread('image006.png');
% figure;
% imshow(inputImage)
segIm = vesselSegPC(inputImage);   
% figure;
% imshow(segIm);
%validation(goundTruth,segIm);
segIm=~segIm;
% figure;
% imshow(segIm);

bwarea=bwareaopen(segIm,55);
% figure;
% imshow(bwarea);

imD=imdilate(bwarea,strel('disk',1));
% figure;
% imshow(imD);

imF=imfill(imD,'holes');
% figure;
% imshow(imF);
end
% imD=~imD;
% figure;
% imshow(imD);


