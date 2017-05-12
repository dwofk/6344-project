A folder containing image results.

inputs/ contains the original, simple image inputs.  These are drawn from the EMPA dataset.

net_outputs/ contains the outputs directly from the neural nets.  These nets were trained to estimate the image produced by a 4x longer exposure period (plus) and a 4x shorter exposure period (minus).  They are further divided into nets with a primary convolution kernel of 1x1 or 3x3.

expfusion/ contains the images produced by processing the net_outputs/ images with exposure fusion in MATLAB (see script exposure_cnn/fuseExposures.m)

makehdr/ contains the images produced by processing the net_outputs/ images with makeHDR in MATLAB (see script exposure_cnn/useMakeHDR.m)

These are further divided into 'default' and 'manual'; default refers to the default MATLAB functions for doing this and manual to a tweaked way.  Both are present in the MATLAB scripts.
