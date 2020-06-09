function localBinaryPatternImage1 = LBP_fn(grayImage)


[rows columns] = size(grayImage);
localBinaryPatternImage1 = zeros(size(grayImage), 'uint8');

for row = 2 : rows - 1
	for col = 2 : columns - 1
		centerPixel = grayImage(row, col);
		pixel7=grayImage(row-1, col-1) >= centerPixel;
		pixel6=grayImage(row-1, col) >= centerPixel;
		pixel5=grayImage(row-1, col+1) >= centerPixel;
		pixel4=grayImage(row, col+1) >= centerPixel;
		pixel3=grayImage(row+1, col+1) >= centerPixel;
		pixel2=grayImage(row+1, col) >= centerPixel;
		pixel1=grayImage(row+1, col-1) >= centerPixel;
		pixel0=grayImage(row, col-1) >= centerPixel;
        
        eightBitNumber = uint8(...
			pixel7 * 2^7 + pixel6 * 2^6 + ...
			pixel5 * 2^5 + pixel4 * 2^4 + ...
			pixel3 * 2^3 + pixel2 * 2^2 + ...
			pixel1 * 2 + pixel0);
        
        localBinaryPatternImage1(row, col) = eightBitNumber;
    end
end

localBinaryPatternImage1(1, :) = localBinaryPatternImage1(2, :);
localBinaryPatternImage1(end, :) = localBinaryPatternImage1(end-1, :);
localBinaryPatternImage1(:, 1) = localBinaryPatternImage1(:, 2);
localBinaryPatternImage1(:, end) = localBinaryPatternImage1(:, end-1);

% figure;
% imshow(localBinaryPatternImage1, []);
% title('Local Binary Pattern');
% hp = impixelinfo();
% hp.Units = 'normalized';
% hp.Position = [0.2, 0.5, .5, .03];

% subplot(2,2,4);
% [pixelCounts, GLs] = imhist(uint8(localBinaryPatternImage1(2:end-1, 2:end-1)));
% bar(GLs, pixelCounts, 'BarWidth', 1, 'EdgeColor', 'none');
% grid on;
% title('Histogram of Local Binary Pattern');

end


