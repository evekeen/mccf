
%---------------------------------------------------
%   Learning MUlti-Channel Correlation Filters for Eye Detection
%   Multi-Channel Correlation Filters : ICCV'13
%   author    : Hamed Kiani
%   date      : 30 June 2014
%---------------------------------------------------

clear all; close all; clc;

%---------------------------------------------------
%   adding helper functions and images path.
%   we use multiPie face dataset for both training and testing, including
%   902 images of frontal faces with slight lighting variations.
%   the first 500 for training and the rest for testing.
%---------------------------------------------------

addpath('helper functions/');
imgsPath = 'kolya-frames/';
im_sz      = [720 1280];
% im_sz      = [128 128];
% imgsPath = 'multiPie Face Dataset/';
imgs     = dir(fullfile(imgsPath, '*.jpg'));

%---------------------------------------------------
%   parameters setting
%---------------------------------------------------

%   The "right eye" is located at [40 96].
%   We normalized and aligned all face images to the size of [128 128], 
%   where the left and right eye are located at [40 32] and [40 96],
%   respectively.

% target_poss = [
%     342.0, 592.67;
%     327, 578.99998;
%     313.67, 569.33;
%     299.33, 561.33;
%     286.33, 553.33
% ];

target_poss = [
    360, 613
    343, 594;
    327, 580;
    312, 570;
    299, 562
];

%   HoG parameters. Pleae refer to "calc_hog" function for more details.
nbins      = 5;
cell_size  = [6 6];
block_size = [3 3];

%   MCCF Gaussian sigma and lambda. Please refer to the reference paper for
%   more details.
sigma  = 2;
lambda = 0.1;

%   Cosine window applied on images to reduce the high frequenceis
%   caused by image borders.
%   Please refer to "get_cosine_window" function.
cos_window = get_cosine_window(im_sz,2);


%---------------------------------------------------
%   MCCF Learning Loops over 500 training face images in two steps
%   (1) computing auto- and cross- correlation energies, xxF and xyF
%   (2) closed-form filter learning
%---------------------------------------------------


%   (1) computing auto- and cross- correlation energies, xxF and xyF

for i = 1:5
    
    %   loading images
    im = imread([imgsPath imgs(i).name]);
%     imshow(im);
%     pause;
    target_pos = target_poss(i,:);
    
    %   RGB to Gray
     if size(im,3) == 3
         gray = double(rgb2gray(im));
     else 
         gray = im;
     end
    
    %   image power-normalization to have zero mean and unit variance
    %   to make MCCF robust against lighting variations
    nor_im = powerNormalise(double(im));
    
    %   desired correlation response for the right eye, a Gaussian-like
    %   output with a high peak located at the right eye [40 96]. The
    %   bandwidth of the Gaussian function is defined by "sigma".
    %   target_pos indicates the location of the target in the training
    %   image.
    corr_rsp = gaussian_filter(size(gray),sigma, target_pos);
    
    %   calculating dense HoG for the normalized face image. nbins,
    %   cell_size and block_size are the HoG parameters.
    %   Please refer to "calc_hog" for more details.
%     hgs = calc_hog(powerNormalise(double(gray)), nbins, cell_size, block_size);
    hogs = calc_hog(nor_im, nbins, cell_size, block_size);
%     [hogs, visualization] = extractHOGFeatures(nor_im, 'CellSize', [4 4], 'BlockSize', [8 8], 'NumBins', 5);
    
%     subplot(1,2,1);
%     imshow(nor_im);
%     subplot(1,2,2);
%     plot(visualization);
%     
%     pause;
    
    %   applying cosine window on the HoG channels to reduce the
    %   high frequencies of image borders.
    hogs = bsxfun(@times, hogs, cos_window);
    
    
    %   FFT2 of HoG feature channels.
    %   "_f" postfix indicates the variable in the frequncy domain.
    hogs_f = fft2(hogs);
    
    %   FFT2 of corr_rsp.
    corr_rsp_f = reshape(fft2(corr_rsp), [],1);
    
    %   diag of HoG channels, explanations in the paper...
    diag_hogs_f = spdiags(reshape(hogs_f, prod(im_sz), []), ...
        [0:nbins-1]* prod(im_sz), prod(im_sz), prod(im_sz)*(nbins));
    
    
    %   Auto- and Cross- correlation energies, xxF and xyF.
    if i==1
        xxF = diag_hogs_f'*diag_hogs_f  ;
        xyF = diag_hogs_f'*corr_rsp_f ;
    else
        xxF = xxF + (diag_hogs_f'*diag_hogs_f);
        xyF = xyF + (diag_hogs_f'*corr_rsp_f);
    end;
    
end;

%   (2) closed-form filter learning

%   I : identity matrix
I     = speye(size(xxF,1));
filtF = (xxF + I*lambda)\xyF;
filtF = (reshape(filtF, im_sz(1), im_sz(2), []));

%   IFFT2
filt  = real(ifft2(filtF));

% circular shift to get non-shifted filters
filt  = circshift(filt, floor(im_sz/2));

%   saving the MCCF filte to use for testing
save ('filt','filt');

% visualizing the learning multi-channel CF
for j = 1: nbins
    subplot(2,3,j) ;imagesc(real(filt(:,:,j)));colormap gray;
    axis off; axis image; title(['MCCF Channel # : ' num2str(j)]);
end;

%%%%%%%%%%%%%%% End of the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
