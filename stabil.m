clear all; close all; clc;

MAX_FEATURES = 10000;
startFrame = 65;

addpath('helper functions/');
imgsPath = 'kolya2/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

% profile on

prev = imread([imgsPath imgs(startFrame).name]);
% prev = rgb2gray(imread([imgsPath imgs(startFrame).name]));
% prev = imcrop(prev, [300, 500, 300, 300]);
orig = prev;
prevGray = rgb2gray(prev);
% ptsPrev = detectORBFeatures(prev);
% strongestPrev = selectStrongest(ptsPrev, MAX_FEATURES);
% [featuresPrev, validPtsPrev] = extractFeatures(prev, strongestPrev, 'Method', 'ORB');

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 200;

outputVideo = VideoWriter(fullfile('.', 'stabilized.avi'));
outputVideo.FrameRate = 30;
open(outputVideo)
writeVideo(outputVideo, prev)

for i = startFrame+1:startFrame+100
    im = imread([imgsPath imgs(i).name]);
    gray = rgb2gray(im);
%     im = rgb2gray(imread([imgsPath imgs(i).name]));
%     im = imcrop(im, [300, 500, 300, 300]);
%     
%     pts  = detectORBFeatures(im);        
%     strongest = selectStrongest(pts, MAX_FEATURES);
%     [features, validPts] = extractFeatures(im, strongest, 'Method', 'ORB');                    
% %     
%     indexPairs = matchFeatures(features, featuresPrev);    
%     matched  = validPts(indexPairs(:,1));   
%     matchedPrev = validPtsPrev(indexPairs(:,2));         
%     
%     [tform, inlierIdx] = estimateGeometricTransform2D(matched, matchedPrev, 'similarity');
%     inlierPts = matched(inlierIdx,:);
%     inlierPtsprev  = matchedPrev(inlierIdx,:);
    
    tform = imregtform(gray, prevGray, 'affine', optimizer, metric);    

    outputView = imref2d(size(prev));
    Ir = imwarp(im, tform, 'OutputView', outputView);
    writeVideo(outputVideo, Ir)
    
      figure
%       imshow(Ir - orig);
      imshowpair(orig, Ir, 'Scaling', 'joint')
%       showMatchedFeatures(Ir, prev, inlierPts, inlierPtsprev)
      title('Matched Inlier Points')     
%       pause;
    
%     prev = im;
%     ptsPrev = pts;
%     featuresPrev = features;
%     validPtsPrev = validPts;
%     strongestPrev = strongest;
end  

close(outputVideo)
