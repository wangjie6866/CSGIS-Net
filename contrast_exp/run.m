clc;
clear;

fileDir = '' ;% input dir
outputDir = ''; % output
inDir = dir(fileDir);
len = length(inDir);

for i=3:len
    path_in = strcat(fileDir,inDir(i).name);
    I = im2double(imread(path_in));
    [m,n,c] = size(I);
    %I = I(:,n/2+1:n,:);
    % RGF
    sigma_s = 4;
    sigma_r = 0.1;
    iteration = 4;
    GaussianPrecision = 0.05;
    tic;
    result_RGF = RollingGuidanceFilter(I, sigma_s, sigma_r, iteration, GaussianPrecision);
    %GF
    result_GF = imguidedfilter(I);
    %L0
    result_L0 = L0Smoothing(I,0.01);
    %RTV
    result_RTV = tsmooth(I,0.015,3);
    toc;
    out_RGF = strcat(outputDir,'RGF/',inDir(i).name);
    imwrite(result_RGF, out_RGF);
    out_GF = strcat(outputDir,'GF/',inDir(i).name);
    imwrite(result_GF, out_GF);
    out_L0 = strcat(outputDir,'L0/',inDir(i).name);
    imwrite(result_L0, out_L0);
    out_RTV = strcat(outputDir,'RTV/',inDir(i).name);
    imwrite(result_RTV, out_RTV);
    %print('ok');
end

