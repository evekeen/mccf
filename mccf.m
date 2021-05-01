clear all; close all; clc;

addpath('helper functions/');
imgsPath = 'kolya1-short-stabil/';
imgs     = dir(fullfile(imgsPath, '*.PNG'));

channels  = 3;
sigma     = 2;
lambda    = 0.1;
proc_time = 0;

img = imread([imgsPath imgs(1).name]);

roi = [576.16,336.52,35.39,11.88];
w = roi(3);
h = roi(4);
WW = 199;
HH = 199;
dw = (WW - w) / 2;
dh = (HH - h) / 2;
rect = [roi(1) - dw, roi(2) - dh, WW, HH];
% center = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];
center = [rect(4)/2 rect(3)/2];

img = imcrop(img, rect);
im_sz = size(img);
im_sz = im_sz(1:2);

model = Model(im_sz);
[filt_f, filt] = model.train(img, center, 40);

for j = 1: channels
    subplot(2,3,j) ;imagesc(real(filt(:,:,j)));colormap gray;
    axis off; axis image; title(['MCCF Channel # : ' num2str(j)]);
end
for j = 1: channels
    subplot(2,3,j + 3) ;imagesc(img(:,:,j)); colormap gray;
    axis off; axis image; title('Orig');
end

pause(0.3);
prevPos = [rect(1) + WW / 2, rect(2) + HH / 2];
vv = [];
pos = [];
T = 1;

%   testing loop starts here!
for i = 2:40    
    tic;
    img = imread([imgsPath imgs(i).name]);
    img = imcrop(img, rect);
    im = powerNormalise(double(img));
%     im = bsxfun(@times, im, cos_window);
    im_sz = size(im);
    im_sz = im_sz(1:2);

    hogs_f = fft2(im);
    rsp_f  = sum(hogs_f.*filt_f,3);    
    rsp    = circshift(real(ifft2(rsp_f)), -im_sz/2);
    
    [y x] = find(rsp == max(max(rsp)));
    
    % ---- more training --------
    [filt_f, filt] = model.train(img, [x, y], 1);    
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
    pause(0.3);
    if i > 2 && diff(1) > 5
        disp(diff)
        pause
    end
end
% close aviobj;
