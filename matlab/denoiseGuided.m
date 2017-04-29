input_filename = 'cabin.jpg';
exp_bracket_path = fullfile(cd, 'exp_bracket_png');

assert(~exist(exp_bracket_path, 'dir') == 0, ...
    'Exposure bracket directory does not exist.');

files = dir([exp_bracket_path, '/exp*']);
num_files = length(files);

if (num_files == 0)
    error('No exposure bracket files found.');
end

% read original image
orig_img = imread(fullfile(cd, 'data', input_filename));
%figure, imshow(orig_img);

% convert original image to YCbCr color space
orig_img_YCbCr = rgb2ycbcr(orig_img);

nhoodSize = 5;
%smoothValue  = 0.001*diff(getrangefromclass(orig_imgB)).^2; % = 65
smoothValue = 5;

save_path = [exp_bracket_path, '_dn'];
if (exist(save_path, 'dir'))
    delete([save_path, '/*']);
else
    mkdir(save_path);
end

for f=1:size(files, 1)
    fname = files(f).name;
    fprintf('Reading file: %s \n', [exp_bracket_path, '/', fname]);
    
    exp_img = imread([exp_bracket_path, '/', fname]);
    %figure, imshow(exp_img);
    
    exp_img_YCbCr = rgb2ycbcr(exp_img);
    
    exp_img_YCbCr(:,:,1) = imguidedfilter( ...
        exp_img_YCbCr(:,:,1), orig_img_YCbCr(:,:,1), ...
        'NeighborhoodSize', nhoodSize, ...
        'DegreeOfSmoothing', smoothValue);
    
    out_img_RGB = ycbcr2rgb(exp_img_YCbCr);
    %figure, imshow(out_img_RGB);
    
    out_fname = [exp_bracket_path, '_dn/', fname];
    imwrite(out_img_RGB, out_fname);
end