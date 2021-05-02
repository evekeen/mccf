classdef Model < handle 
    properties
        initialized = false
        xxF
        xyF
        cos_window
        
        sigma  = 1
        lambda = 0.1
    end
    methods
        function obj = Model(im_sz)
            obj.cos_window = get_cosine_window(im_sz, 2);
        end                        
        
        function [filt_f, filt] = train(obj, img, center, iterations)
            channels  = 3;                      
            im_sz = size(img);
            im_sz = im_sz(1:2);

            for i = 1:iterations      
                im = rand_warp(img);
                im = powerNormalise(double(im));                

                corr_rsp = gaussian_filter(im_sz, obj.sigma, center);
                im = bsxfun(@times, im, obj.cos_window);
                im_f = fft2(im);
                corr_rsp_f = reshape(fft2(corr_rsp), [],1);
                diag_hogs_f = spdiags(reshape(im_f, prod(im_sz), []), [0:channels-1]* prod(im_sz), prod(im_sz), prod(im_sz)*(channels));
                if obj.initialized == false
                    obj.initialized = true;
                    obj.xxF = diag_hogs_f'*diag_hogs_f;
                    obj.xyF = diag_hogs_f'*corr_rsp_f;
                else
                    obj.xxF = obj.xxF + (diag_hogs_f'*diag_hogs_f);
                    obj.xyF = obj.xyF + (diag_hogs_f'*corr_rsp_f);
                end
            end
            
            [filt_f, filt] = obj.getFilter(im_sz);
        end
        
        function [filt_f, filt] = getFilter(obj, im_sz)
            I     = speye(size(obj.xxF,1));
            filtF = (obj.xxF + I*obj.lambda)\obj.xyF;
            filtF = (reshape(filtF, im_sz(1), im_sz(2), []));
            filt  = real(ifft2(filtF));
            filt  = circshift(filt, floor(im_sz/2));
            filt_f = fft2(filt);
        end
    end
end