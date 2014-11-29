% Princu Jain (Reg. No.: 201230141) and Kalpit Thakkar (Reg. No.: 201201071), November 2014 %
% Project Title : Pool Table Edge, Pocket and Ball Position Estimation, for Cue Guiding %
% Digital Image Processing (CSE478) - Course Project %
% International Institute Of Information Technology, Hyderabad %

% Image Processing Techniques used : %

%% Function that takes in a top-view image of a pool table and processes it.

function poolCueGuider(pictureType)

close all; clear all;

% Read the image according to the type mentioned
if pictureType == 'Sun':
    I = imread('Sun.JPG');
else if pictureType == 'Artificial':
        I = imread('Artificial.JPG');
    else :
        fprintf('Incorrect input arguments!\n Usage : poolCueGuider <arg>\n where, arg = Sun/Artificial\n');
    end
end

grayImage = rgb2gray(I);
doubleImage = im2double(I);

figure;
title('Original Image');
imshow(doubleImage);

%% For Binarizing the image

[height width] = size(doubleImage);
redComponent = doubleImage(:, :, 1);
greenComponent = doubleImage(:, :, 2);
blueComponent = doubleImage(:, :, 3);
magnitudeImage = (redComponent.^2 + greenComponent.^2 + blueComponent.^2).^0.5;

normRed = redComponent./magnitudeImage;
normGreen = greenComponent./magnitudeImage;
normBlue = blueComponent./magnitudeImage;

[redCutoff, greenCutoff, blueCutoff] = findCutoffPoints(height, width, normRed, normGreen, normBlue);

binThresh = redCutoff.*normRed + greenCutoff.*normGreen + blueCutoff.*normBlue;
binarizedImage = binThresh > 0.98;

figure;
title('Binarized Image using weight vector');
imshow(binarizedImage);

%% OTSU

%% Region Labelling and final segmented table image

% Labelling regions and region properties
labeledImage = bwlabel(binarizedImage, 4);
blobProperties = regionprops(labeledImage, I_gray, 'all');

blobArea = cat(1, blobProperties.Area);
blobCentroid = cat(1, blobProperties.Centroid);
blobBoundingbox = cat(1, blobProperties.BoundingBox);

[~, largestBlob] = max(blobArea);  % Get largest region %

% Find centroid and Bounding Box of the table region
tableCentroid = blobCentroid(largestBlob, :);
tableBound = blobBoundingbox(largestBlob, :, :, :, :);
finalImage = binarizedImage & (labeledImage == largestBlob);

