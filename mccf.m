clear all; close all; clc;

addpath('helper functions/');
imgsPath = 'kolya1-short-stabil/';
roi = [576.16,336.52,35.39,11.88];

% imgsPath = 'kolya2-late-stabil/';
% roi = [531.05,645.57,14.16,4.45];

imgs     = dir(fullfile(imgsPath, '*.PNG'));

channels  = 3;
proc_time = 0;

img = imread([imgsPath imgs(1).name]);

w = roi(3);
h = roi(4);
WW = floor(roi(3) * 1.8);
if rem(WW, 2) == 0
    WW = WW + 1;
end
HH = WW;
dw = (WW - w) / 2;
dh = (HH - h) / 2;
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
pause
prevPos = [rect(1) + WW / 2, rect(2) + HH / 2];
vv = [];
pos = [];
T = 1;

origSigma = model.sigma;

%   testing loop starts here!
for i = 2:100    
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
    
    % ---- more training --------
    model.sigma = origSigma - i * 0.01;
    [filt_f, filt] = model.train(img, [x, y], 50);    
    %----------------------------
    
    newAbs = [rect(1) + x, rect(2) + y];
    if i == 2
        v = newAbs - prevPos;
        initialState = [
            rect(1) + center(1); v(1); 0;
            rect(2) + center(2); v(2); 0
        ];
        KF = trackingKF('MotionModel', '2D Constant Acceleration', 'State', initialState);
        predict(KF, T);            
    else        
        diff = newAbs - [predicted(1) predicted(4)];        
    end      

    correct(KF, newAbs);
    predicted = predict(KF, T);    
    
    rect = [predicted(1) - WW / 2, predicted(4) - HH / 2, WW, HH];

    proc_time = proc_time + toc;
    subplot(1,2,1);
    imagesc(rsp); colormap gray;axis image ; axis off;title ('Correlation rsp.');
    
    subplot(1,2,2);
    imagesc(img); colormap gray;axis image ; axis off;title ('image');             
%     hold on; plot(96,40, 'ob','MarkerSize',10,'LineWidth',3);
    hold on; plot(x, y, '*r','MarkerSize',10,'LineWidth',2);
    pause(0.01);
    if i > 2 && diff(1) > 8
        disp(diff)
%         pause
    end
end
% close aviobj;
