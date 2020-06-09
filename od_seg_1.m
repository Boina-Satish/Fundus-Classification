function bw = od_seg_1(im,i) 

imR = im(:,:,1); %Red Channel Extraction

%%
[rows columns] = size(imR);
clippedImage = imR; % Initialize.
clippedImage(:, 1:0.6*columns) = imR(1,1);
clippedImage(1:0.2*rows,:) = imR(1,1);
clippedImage(0.7*rows:end,:) = imR(1,1);

% figure, imshow(clippedImage);
% hp = impixelinfo();
% hp.Units = 'normalized';
% hp.Position = [0.2, 0.5, .5, .03];

J = medfilt2(clippedImage,'symmetric');
% figure, imshow(J);

%contrast stretching
J = J.*0.5;
% figure, imshow(J);
% hp = impixelinfo();
% hp.Units = 'normalized';
% hp.Position = [0.2, 0.5, .5, .03];

% Binarize the image
binaryImage = J >= 102;
% figure, imshow(binaryImage);
% Take largest blob
binaryImage = bwareafilt(binaryImage, 1);
% figure, imshow(binaryImage);

binaryImage = imclearborder(binaryImage,1);
% figure, imshow(binaryImage);
% % Take convex hull
% binaryImage = bwconvhull(binaryImage);
% figure, imshow(binaryImage);

bw = ~binaryImage;
% figure, imshow(binaryImage);
