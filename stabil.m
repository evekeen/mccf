clear all; close all; clc;

MAX_FEATURES = 1000;
startFrame = 61;

addpath('helper functions/');
imgsPath = 'kolya2/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

prev = rgb2gray(imread([imgsPath imgs(startFrame).name]));
ptsPrev = detectORBFeatures(prev);
strongestPrev = selectStrongest(ptsPrev, MAX_FEATURES);
[featuresPrev, validPtsPrev] = extractFeatures(prev, strongestPrev);


for i = startFrame+1:500
    im = rgb2gray(imread([imgsPath imgs(i).name]));
    
    pts  = detectORBFeatures(im);        
    strongest = selectStrongest(pts, MAX_FEATURES);
    [features, validPts] = extractFeatures(im, strongest);                    
    
    indexPairs = matchFeatures(features, featuresPrev);    
    matched  = validPts(indexPairs(:,1));   
    matchedPrev = validPtsPrev(indexPairs(:,2));
    
%     figure
%     showMatchedFeatures(im, prev, matched, matchedPrev)
%     title('Candidate matched points (including outliers)')
    
    [tform, inlierIdx] = estimateGeometricTransform2D(matched, matchedPrev, 'similarity');
    inlierPts = matched(inlierIdx,:);
    inlierPtsprev  = matchedPrev(inlierIdx,:);
    
    figure
    showMatchedFeatures(im, prev, inlierPts, inlierPtsprev)
    title('Matched Inlier Points')
    
    pause;
    
    prev = im;
    ptsPrev = pts;
    featuresPrev = features;
    validPtsPrev = validPts;
    strongestPrev = strongest;
end
