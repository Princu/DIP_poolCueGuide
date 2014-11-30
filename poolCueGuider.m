% Princu Jain (Reg. No.: 201230141) and Kalpit Thakkar (Reg. No.: 201201071), November 2014 %
% Project Title : Pool Table Edge, Pocket and Ball Position Estimation, for Cue Guiding %
% Digital Image Processing (CSE478) - Course Project %
% International Institute Of Information Technology, Hyderabad %

% Image Processing Techniques used : Morphological Operations, Histogram Thresholding, Region labeling and Segmentation, Template Matching %
                                  
%% Function that takes in a top-view image of a pool table and processes it.

function[] = poolCueGuider(pictureType)

input = strcmp(pictureType, 'Sun');
% Read the image according to the type mentioned
if input
    I = imread('Sun.JPG');
else
    I = imread('Artificial.JPG');
end

grayImage = rgb2gray(I);
doubleImage = im2double(I);

figure('name', 'Original Image');
imshow(doubleImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

%% Binarizing Code

[height width ~] = size(doubleImage);
redComponent = doubleImage(:, :, 1);
greenComponent = doubleImage(:, :, 2);
blueComponent = doubleImage(:, :, 3);
magnitudeImage = (redComponent.^2 + greenComponent.^2 + blueComponent.^2).^0.5;

normRed = redComponent./magnitudeImage;
normGreen = greenComponent./magnitudeImage;
normBlue = blueComponent./magnitudeImage;

%% Find the weight vector for binarizing

rowInit = height * 1/4;
rowEnd = height * 3/4;
colInit = width * 1/4;
colEnd = width * 3/4;

poolTableRedComponent = normRed(rowInit : rowEnd, colInit : colEnd);
poolTableGreenComponent = normGreen(rowInit : rowEnd, colInit : colEnd);
poolTableBlueComponent = normBlue(rowInit : rowEnd, colInit : colEnd);
poolTableMagComponent = magnitudeImage(rowInit : rowEnd, colInit : colEnd);

% Find histogram characteristics for each component
[nr, redHist] = imhist(poolTableRedComponent);
[ng, greenHist] = imhist(poolTableGreenComponent);
[nb, blueHist] = imhist(poolTableBlueComponent);
[nm, magHist] = imhist(poolTableMagComponent);

% Find location of values in component histograms
[cr, redPeak] = max(nr);
[cg, greenPeak] = max(ng);
[cb, bluePeak] = max(nb);
[cm, magPeak] = max(nm);

% Locations in the histogram where peaks occur
redCutoff = redHist(redPeak);
greenCutoff = greenHist(greenPeak);
blueCutoff = blueHist(bluePeak);
magCutoff = magHist(magPeak);

binThresh = redCutoff.*normRed + greenCutoff.*normGreen + blueCutoff.*normBlue;
binarizedImage = binThresh > 0.98;

figure('name', 'Binarized image using weight vector');
imshow(binarizedImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

%% Region Labelling and final segmented table image

% Labelling regions and region properties
labeledImage1 = bwlabel(binarizedImage, 4);
blobProperties = regionprops(labeledImage1, grayImage, 'all');

blobArea = cat(1, blobProperties.Area);
blobCentroid = cat(1, blobProperties.Centroid);
blobBoundingbox = cat(1, blobProperties.BoundingBox);

[~, largestBlob] = max(blobArea);  % Get largest region %

% Find centroid and Bounding Box of the table region
tableCentroid = blobCentroid(largestBlob, :);
tableBound = blobBoundingbox(largestBlob, :, :, :, :);
finalImage = binarizedImage & (labeledImage1 == largestBlob);
figure('name', 'Binarized Image (Pockets not marked yet)');
imshow(finalImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

for column = 1 : width
    for row = 1 : height
        if abs(row - tableCentroid(2)) < tableBound(4)/2 && abs(column - tableCentroid(1)) < tableBound(3)/2
            fillTableImage(row, column) = 1;
        else
            fillTableImage(row, column) = 0;
        end
    end
end

figure('name', 'The filled rectangle table');
imshow(fillTableImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

deltaImage = xor(fillTableImage, finalImage);

figure('name', 'Final binarized image');
imshow(deltaImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

%% OTSU

%edgePic = imfill(binarizedImage, 'holes');
%upperLeft = zeros(2,1);
%upperRight = zeros(2,1);
%lowerLeft = zeros(2,1);
%lowerRight = zeros(2,1);

%centroid = blobCentroid(1, :);
%boundingBox = blobBoundingbox(1, :);

%upperLeft 

%% Erode the image, to remove the noise for better detection

degOfErosion = 70;
if tableBound(3) < tableBound(4)
    elementRadius = round(tableBound(3)./degOfErosion);
else
    elementRadius = round(tableBound(4)./degOfErosion);
end

SE = strel('disk', elementRadius);

erodedImage = imerode(deltaImage, SE);

figure('name', 'Eroded Image');
imshow(erodedImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

labeledImage2 = bwlabel(erodedImage, 4);
blobProperties = regionprops(labeledImage2, grayImage, 'all');
blobArea = cat(1, blobProperties.Area);
blobCentroid = cat(1, blobProperties.Centroid);
blobBoundingbox = cat(1, blobProperties.BoundingBox);

dilatedImage = imdilate(erodedImage, SE);
figure('name', 'Dilated Image');
imshow(dilatedImage);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

%% Find the location of the table pockets

% locates positions of pockets by finding objects closest to the vertices of the table bounding box given by region labeling %
vertex(1,1) = tableCentroid(1) - tableBound(3)/2;
vertex(1,2) = tableCentroid(2) - tableBound(4)/2;
vertex(2,1) = tableCentroid(1);
vertex(2,2) = tableCentroid(2) - tableBound(4)/2;
vertex(3,1) = tableCentroid(1) + tableBound(3)/2;
vertex(3,2) = tableCentroid(2) - tableBound(4)/2;
vertex(4,1) = tableCentroid(1) - tableBound(3)/2;
vertex(4,2) = tableCentroid(2) + tableBound(4)/2;
vertex(5,1) = tableCentroid(1);
vertex(5,2) = tableCentroid(2) + tableBound(4)/2;
vertex(6,1) = tableCentroid(1) + tableBound(3)/2;
vertex(6,2) = tableCentroid(2) + tableBound(4)/2;

for pocketIndex = 1:6
    min(pocketIndex) = 1e4;
end

for pocketIndex = 1:6
    for object = 1:length(blobArea)
        vertexDist(pocketIndex) = sqrt((blobCentroid(object, 1) - vertex(pocketIndex, 1)).^2 + (blobCentroid(object, 2) - vertex(pocketIndex, 2)).^2);
        if vertexDist(pocketIndex) < min(pocketIndex) && vertexDist(pocketIndex) < 8 * elementRadius
            min(pocketIndex) = vertexDist(pocketIndex);
            pocket(pocketIndex) = object;
            centroidPocket(pocketIndex, :) = blobCentroid(object, :);
            boundingboxPocket(pocketIndex, :) = blobBoundingbox(object, :);
        end
    end
end

% finds "radius" of pockets by adding eroded objects bounding box widths to the erosion structuring element's radius %
radius = (boundingboxPocket(:, 3) + boundingboxPocket(:, 4))/4 + elementRadius;
[n, xout] = hist(radius);

[maxVal, maxLoc] = max(n);

pocketRadius = xout(maxLoc);

% creates a binarized picture of pockets alone
picturePockets = labeledImage2 * 0;
for pocketIndex = 1:6
    picturePockets = picturePockets | (pocket(pocketIndex) == labeledImage2);
end
picturePockets = imdilate(picturePockets, SE);

figure('name', 'Finding the Pockets!');
imshow(picturePockets);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end

NumberOfPockets = 6;

%% Find the balls location
index = 1;

%Removes pocket from consideration
for object = 1:length(blobArea)
    minDist = 1e4;
    for pocketIndex = 1:6
        distance = sqrt((centroidPocket(pocketIndex,1) - blobCentroid(object,1)).^2 + (centroidPocket(pocketIndex,2) - blobCentroid(object,2)).^2);
        if distance < minDist
            minDist = distance;
        end
    end
    if minDist > 0.8 * pocketRadius %%%%SEE THIS WHY IS THIS 0.8
        ball(index) = object;
        centroidBall(index,:) = blobCentroid(object,:);
        boundingboxBall(index,:) = blobBoundingbox(object,:);
        index = index + 1;
    end
end
numberBalls = length(ball);

radius = (boundingboxBall(:,3) + boundingboxBall(:,4))/4 + elementRadius;
[n, xout] = hist(radius);
[maxValue, maxLocation] = max(n);
ballRadius = xout(maxLocation);
numberBallstemp = numberBalls;
balltemp = ball;
centroidBalltemp = centroidBall;
boundingboxBalltemp = boundingboxBall;

% Removes non-shiny part objects
for object = 1:numberBalls
    rowStart = round(centroidBall((object),2) - ballRadius);
    rowEnd = round(centroidBall((object),2) + ballRadius);
    colStart = round(centroidBall((object),1) - ballRadius);
    colEnd = round(centroidBall((object),1) + ballRadius);
    picBall = doubleImage(rowStart:rowEnd, colStart:colEnd, :);
    picBallMag = sqrt(picBall(:,:,1).^2 + picBall(:,:,2).^2 + picBall(:,:,3).^2);
    [hMag, xoutm] = imhist(picBallMag);
    [maxValue, maxLocation] = max(hMag);
    if maxLocation/length(hMag) < 0.9
        for i = object : numberBalls - 1
            balltemp(i) = balltemp(i+1);
            centroidBalltemp(i,:) = centroidBalltemp(i+1,:);
            boundingboxBalltemp(i,:) = boundingboxBalltemp(i+1,:);
        end
        numberBallstemp = numberBallstemp-1;
    end
end

numberBalls = numberBallstemp;
ball = balltemp;
centroidBall = centroidBalltemp;
boundingboxBall = boundingboxBalltemp;

%Create avg Ball binarized Template
d = 0;

for object= 1:numberBalls
    rowStart = round(centroidBall((object),2)- 1.0*ballRadius);
    rowEnd = rowStart + round(2*1.0*ballRadius);
    colStart = round(centroidBall((object),1) - 1.0*ballRadius);
    colEnd = colStart + round(2*1.0*ballRadius);
    if d == 0
        picBall_Bin = deltaImage(rowStart:rowEnd,colStart:colEnd);
        d = 1;
    else
        picBall_Bin = picBall_Bin + deltaImage(rowStart:rowEnd,colStart:colEnd);
    end
end
picBall_Bin = round(picBall_Bin./numberBalls);

% Find most likely center (x,y) of each ball using template matching

extent = round(ballRadius/8);
for object = 1 : numberBalls
    picBall_Max = 0;
    rowOff = 0;
    colOff = 0;
    for i = -extent : extent
        for j = -extent : extent
            rowStart = round(centroidBall((object),2) - 1.0*ballRadius + i);
            rowEnd = rowStart + round(2*1.0*ballRadius);
            colStart = round(centroidBall((object),1) - 1.0*ballRadius + j);
            colEnd = colStart + round(2*1.0*ballRadius);
            picBall_Temp = deltaImage(rowStart:rowEnd, colStart:colEnd);
            picBall_Sum = sum(sum(picBall_Bin & picBall_Temp));
            if picBall_Sum > picBall_Max
                picBall_Max = picBall_Sum;
                rowOff = i;
                colOff = j;
            end
        end
    end
    centroidBall(object,2) = centroidBall(object,2) + rowOff;
    centroidBall(object,1) = centroidBall(object,1) + colOff;
end
pictureBalls = labeledImage2 * 0;
for ballIndex = 1:numberBalls
    pictureBalls = pictureBalls | (ball(ballIndex) == labeledImage2);
end
pictureBalls = imdilate(pictureBalls, SE);

figure('name', 'Finding the Balls!');
imshow(pictureBalls);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end


%% Find the Cue Ball (white ball) and the Eight Ball (black ball)

wMax = 0;
wMin = 99999;

index = 1;

for object = 1:numberBalls
    
    rowInit = round(centroidBall((object), 2) - 1.0 * ballRadius);
    rowEnd = rowInit + round(2 * 1.0 * ballRadius); 
    colInit = round(centroidBall((object), 1) - 1.0 * ballRadius);
    colEnd = colInit + round(2 * 1.0 * ballRadius);
    picBallBin = deltaImage(rowInit : rowEnd, colInit : colEnd);
    picBall = doubleImage(rowInit : rowEnd, colInit : colEnd, :);
    picBallMag = sqrt(picBall(:,:,1).^2 + picBall(:,:,2).^2 + picBall(:,:,3).^2);
    picBallRed = picBall(:,:,1)./picBallMag.*picBallBin;
    picBallGreen = picBall(:,:,2)./picBallMag.*picBallBin;
    picBallBlue = picBall(:,:,3)./picBallMag.*picBallBin;
    picBallMag = picBallMag.*picBallBin;
    
    [hRed, xoutr] = imhist(picBallRed);
    [hGreen, xoutg] = imhist(picBallGreen);
    [hBlue, xoutb] = imhist(picBallBlue);
    [hMag, xoutm] = imhist(picBallMag);
    
    hRed(1) = 0;
    hGreen(1) = 0;
    hBlue(1) = 0;
    hMag(1) = 0;
    
    scale = length(hRed)/2;
    startPos = round(scale./xoutr(scale) * 0.5);
    endPos = round(scale./xoutr(scale) * 0.7);
    startPos2 = round(0.9 * length(hRed));
    endPos2 = length(hRed);
    
    % integrates histogam values around white values
    redTop = sum(hRed(startPos : endPos));
    greenTop = sum(hGreen(startPos : endPos));
    blueTop = sum(hBlue(startPos : endPos));
    magTop = sum(hMag(startPos2 : endPos2));
    
    % for integrated white-region histogram values, selects ball with largest integrates values as cue ball; % 
    % obtains white balance multipliers from cue ball pixel color values %
    if (redTop + greenTop + blueTop + magTop) > wMax
        wMax = (redTop + greenTop + blueTop + magTop);
        cueBall = object;
        centroidCue(1, :) = centroidBall(object, :);
        [rows, cols] = size(picBallRed);
        for row = 1:rows
            for col = 1:cols
                temp = max(picBallRed(row, col), picBallGreen(row, col));
                temp = max(temp, picBallBlue(row, col));
                if temp ~= 0
                    multMat(row, col, 1) = temp./picBallRed(row, col);
                    multMat(row, col, 2) = temp./picBallGreen(row, col);
                    multMat(row, col, 3) = temp./picBallBlue(row, col);
                else
                    multMat(row, col, 1) = 1;
                    multMat(row, col, 2) = 1;
                    multMat(row, col, 3) = 1;
                end
            end
        end
    end
    
    redBot = sum(hRed(startPos : endPos));
    greenBot = sum(hGreen(startPos : endPos));
    blueBot = sum(hBlue(startPos : endPos));
    magBot = sum(hMag(startPos2 : endPos2));
    
    % selects darkest object as eight ball
    if magBot < wMin && cueBall ~= object && (redTop + greenTop + blueTop)./(sum(hRed) + sum(hGreen) + sum(hBlue)) > 0.3
        wMin = magBot;
        eightBall = object;
        centroidEight(1, :) = centroidBall(object, :);
    end
end
numberEight = 1;

%% Find the striped and solid balls

pictureMarked = doubleImage;

numberSolids = 1;
numberStripes = 1;
index = 1;

for object = 1:numberBalls
    
    rowInit = round(centroidBall((object), 2) - 1.0 * ballRadius);
    rowEnd = rowInit + round(2 * 1.0 * ballRadius);
    colInit = round(centroidBall((object), 1) - 1.0 * ballRadius);
    colEnd = colInit + round(2 * 1.0 * ballRadius);
    picBallBin = deltaImage(rowInit : rowEnd, colInit : colEnd);
    picBall = doubleImage(rowInit : rowEnd, colInit : colEnd, :);
    
    % white balances each ball
    picBall = picBall.*multMat;
  
    % inserts white-balanced ball image into marked picture
    pictureMarked(rowInit : rowEnd, colInit : colEnd, :) = picBall;
    picBallMag = sqrt(picBall(:, :, 1).^2 + picBall(:, :, 2).^2 + picBall(:, :, 3).^2);
    picBallRed = picBall(:,:,1)./picBallMag.*picBallBin;
    picBallGreen = picBall(:,:,2)./picBallMag.*picBallBin;
    picBallBlue = picBall(:,:,3)./picBallMag.*picBallBin;

    % inner product with white weight vector
    weightedProd = 0.57 * picBallRed + 0.57 * picBallGreen + 0.57 * picBallBlue;
    picBallTemp = weightedProd > 0.96 & picBallMag > 0.9;
    
%    [rows, cols] = size(picBall);
    
    % those >35% white are labeled stripes, otherwise solids
    whiteFrac = sum(sum(picBallTemp))./(pi * ballRadius.^2);
    if whiteFrac < 0.35 && object ~= cueBall && object ~= eightBall
        solidBall(numberSolids) = object;
        centroidSolid(numberSolids,:) = centroidBall(object, :);
        numberSolids = numberSolids + 1;
    elseif object ~= cueBall && object ~= eightBall
        stripeBall(numberStripes) = object;
        centroidStripe(numberStripes, :) = centroidBall(object, :);
        numberStripes = numberStripes + 1;
    end
end

%% Show the final image with striped, solids, cue and black ball

% Creates colored rings around balls, corresponding to ball radius and centroid; broken rings mark stripes, solid rings mark solids %

extent1 = round(3 * ballRadius/40);
extent2 = round(10 * ballRadius/40);
for i = 1:numberSolids-1
    for theta = 0:360
        col = round(centroidSolid(i, 1) + ballRadius.*cos(theta * pi/180));
        row = round(centroidSolid(i, 2) + ballRadius.*sin(theta * pi/180));
        pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 1) = 0.2;
        pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 2) = 0.6;
        pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 3) = 1;
    end
end

for i = 1:numberStripes-1
    for theta = 0:360
    %    if abs(cos(theta*pi/180)) > 0.9 || abs(sin(theta*pi/180)) > 0.9
            col = round(centroidStripe(i, 1) + ballRadius.*cos(theta * pi/180));
            row = round(centroidStripe(i, 2) + ballRadius.*sin(theta * pi/180));
            pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 1) = 0.6;
            pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 2) = 0.8;
            pictureMarked(row - extent1 : row + extent1, col - extent1 : col + extent1, 3) = 0.2;
    %    end
    end
end

% marks center of each pocket
for object = 1:6
    row = centroidPocket(object, 2);
    col = centroidPocket(object, 1);
    pictureMarked(row - 2*extent2 : row + 2*extent2, col - 2*extent2 : col + 2*extent2, 1) = 1;
    pictureMarked(row - 2*extent2 : row + 2*extent2, col - 2*extent2 : col + 2*extent2, 2) = 1;
    pictureMarked(row - 2*extent2 : row + 2*extent2, col - 2*extent2 : col + 2*extent2, 3) = 1;
end

figure('name', 'Final Image');
imshow(pictureMarked);
w = waitforbuttonpress;
if w == 0
    disp('Button click')
    exit(0);
else
    disp('Key press')
    exit(0);
end