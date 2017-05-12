function ssim_value = computeSSIMColor( X0, X )

    if size(X0, 3) ~= 3 || size(X, 3) ~= 3
        error('X0 and X must be color images!');
    end

    % convert X0 to YCbCr space
    X0_YCbCr = rgb2ycbcr(double(X0));
    X0_Y = X0_YCbCr(:,:,1);
    X0_Cb = X0_YCbCr(:,:,2);
    X0_Cr = X0_YCbCr(:,:,3);

    % convert X to YCbCr space
    X_YCbCr = rgb2ycbcr(double(X));
    X_Y = X_YCbCr(:,:,1);
    X_Cb = X_YCbCr(:,:,2);
    X_Cr = X_YCbCr(:,:,3);

    % mean SSIM values for individual channels
    [SSIM_Y, ~] = ssim(X0_Y, X_Y);
    [SSIM_Cb, ~] = ssim(X0_Cb, X_Cb);
    [SSIM_Cr, ~] = ssim(X0_Cr, X_Cr);

    % compute weighted SSIM value
    ssim_value = (3/4)*SSIM_Y + (1/8)*SSIM_Cb + (1/8)*SSIM_Cr;
    
end