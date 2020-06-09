function BW = od_seg(imR)


%%take red channel and process it
imR = imR(:,:,1);
% figure;
% imshow(imR);
% title('candidate for disk extrxaction');
% hp = impixelinfo();
% hp.Units = 'normalized';
% hp.Position = [0.2, 0.5, .5, .03];

%%contrast stretching
imC = contr_man(imR);
% figure, imshow(imC),title('Image after Contrast Stretching');
% hp = impixelinfo();
% hp.Units = 'normalized';
% hp.Position = [0.2, 0.5, .5, .03];

% %%Thresholding
imT = imC>100;
% figure, imshow(imT);

%%Morphological operations
imD=imerode(imT,strel('disk',5));
imD=imdilate(imT,strel('disk',2));

% figure;
% imshow(imD);
% title('MOrphological Operations done on disk Image');

BW = bwareaopen(imD,1000);
% figure, imshow(BW);
% title('bwareaopen');
BW = ~BW;

% %%Segmentation
% im1 = double(im);
% im2 = double(BW);
% for i=1:3
%  im3(:,:,i) = im2.*im1(:,:,i);
% end
% imDisk = uint8(im3);
% figure, imshow(imDisk);
% title('Segmented image');
end
