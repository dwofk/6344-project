
% well-exposedness measure (from Mertens exposure fusion)
%
% ************************************************************************
%
% Implementation of Exposure Fusion, as described in:
% 
% "Exposure Fusion",
% Tom Mertens, Jan Kautz and Frank Van Reeth
% In proceedings of Pacific Graphics 2007
% 
% Written by Tom Mertens, Hasselt University, August 2007
% Please contact me via tom.mertens@gmail.com for comments and bugs.
%
% Source code obtained from: https://mericam.github.io
%
% ************************************************************************

function C = well_exposedness(I)
    sig = .2;
    N = size(I,4);
    C = zeros(size(I,1),size(I,2),N);
    for i = 1:N
        R = exp(-.5*(I(:,:,1,i) - .5).^2/sig.^2);
        G = exp(-.5*(I(:,:,2,i) - .5).^2/sig.^2);
        B = exp(-.5*(I(:,:,3,i) - .5).^2/sig.^2);
        C(:,:,i) = R.*G.*B;
    end
end