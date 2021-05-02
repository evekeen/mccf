clear all; close all; clc;

addpath('helper functions/');
imgsPath = 'kolya1-short-stabil/';
roi = [576.16,336.52,35.39,11.88];
roi = [460.46,208.21,12.66,3.55];
start = 19;

% imgsPath = 'kolya2-late-stabil/';
% roi = [531.05,645.57,14.16,4.45];

imgs     = dir(fullfile(imgsPath, '*.PNG'));

channels  = 3;
proc_time = 0;

img = imread([imgsPath imgs(start).name]);

w = roi(3);
h = roi(4);
WW = floor(w * 3);
if rem(WW, 2) == 0
    WW = WW + 1;
end
HH = WW;
dw = (WW - w) / 2;
dh = (HH - h) / 2;
FINE_W = w;
FINE_H = h;

rect = [roi(1) - dw, roi(2) - dh, WW, HH];
% center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];
center = [rect(4)/2 rect(3)/2];

img = imcrop(img, rect);
im_sz = size(img);
im_sz = im_sz(1:2);

model = Model(im_sz);
[filt_f, filt] = model.train(img, center, 200);

for j = 1: channels
    subplot(2,3,j) ;imagesc(real(filt(:,:,j)));colormap gray;
    axis off; axis image; title(['MCCF Channel # : ' num2str(j)]);
end
for j = 1: channels
    subplot(2,3,j + 3) ;imagesc(img(:,:,j)); colormap gray;
    axis off; axis image; title('Orig');
end

pause(0.3);
% pause
prevPos = [rect(1) + WW / 2, rect(2) + HH / 2];
vv = [];
pos = [];
T = 1;

origSigma = model.sigma;
prevDistance = 0;
predLocal = center;

%   testing loop starts here!
for i = start+1:110    
    tic;
    img = imread([imgsPath imgs(i).name]);
    img = imcrop(img, rect);
    im = powerNormalise(double(img));
    im = bsxfun(@times, im, model.cos_window);
    im_sz = size(im);
    im_sz = im_sz(1:2);

    im_f = fft2(im);
    rsp_f  = sum(im_f.*filt_f,3);    
    rsp    = circshift(real(ifft2(rsp_f)), -im_sz/2);
    
    [y x] = find(rsp == max(max(rsp)));
    
    bn = imcrop(imbinarize(im2gray(im), 'adaptive'), [x - FINE_W / 2, y - FINE_H / 2, FINE_W, FINE_H]);
    bw2 = bwareafilt(bn, 1);
    measurements = regionprops(bw2, 'Centroid');
    cx = x - FINE_W / 2 + measurements.Centroid(1) - 1;
    cy = y - FINE_H / 2 + measurements.Centroid(2) - 1;   
    
    newAbs = [rect(1) + cx, rect(2) + cy];
    trainIterations = 40;
    if i == start+1
        v = newAbs - prevPos;
        initialState = [
            rect(1) + center(1); v(1); 0;
            rect(2) + center(2); v(2); 0
        ];
        KF = trackingKF('MotionModel', '2D Constant Acceleration', 'State', initialState);
        predict(KF, T);            
    else        
        diff = newAbs - [predicted(1) predicted(4)];
        distance = (diff(1)^2 + diff(2)^2);
        effectiveDistance = distance + prevDistance * 0.5;
        predLocal = [round(predicted(1)-rect(1)), round(predicted(4)-rect(2))];
%         model.sigma = origSigma * max(1, sqrt(distance));
        if effectiveDistance > 2
            maxResp = max(max(rsp));
            atPredicted = rsp(predLocal(2), predLocal(1));
%             fprintf('>> maxResp %.2f atPredicted %.2f\n', maxResp, atPredicted)
%             model.sigma = origSigma * 2;
%         trainIterations = trainIterations / (effectiveDistance * 2 + 1);
            trainIterations = max(0, 50 - effectiveDistance * 10);
        end
        fprintf('#%d training on %.1f distance %.2f sigma: %.2f\n', i, trainIterations, distance, model.sigma)        
        prevDistance = distance;
    end      
    
    % ---- more training --------
%     model.sigma = origSigma - i * 0.005; 
    [filt_f, filt] = model.train(img, [x, y], trainIterations);    
    %----------------------------

    correct(KF, newAbs);
    predicted = predict(KF, T);    
    
    rect = [predicted(1) - WW / 2, predicted(4) - HH / 2, WW, HH];

    proc_time = proc_time + toc;
    figure;
    subplot(1,2,1);
    imagesc(rsp); colormap gray;axis image ; axis off;title ('Correlation rsp.');
    
    subplot(1,2,2);
    imagesc(img); axis image ; axis off;title ('image');                 
    hold on; plot(predLocal(1), predLocal(2), 'ob','MarkerSize',10,'LineWidth',3);
    hold on; plot(x, y, '*r','MarkerSize',10,'LineWidth',2);
        
%     subplot(1,3,3);
%     imagesc(bn); axis image; axis off; title ('Blob');
    hold on; plot(cx, cy, 'x', 'MarkerSize', 10, 'LineWidth',3);
    
    pause(0.05);
    if i > start+1 && diff(1) > 2
%         disp(diff)
%         pause
    end
end
% close aviobj;
