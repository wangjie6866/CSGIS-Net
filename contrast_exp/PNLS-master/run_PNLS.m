clc;
clear;
parpool(10);
% fileDir = '' ;% input dir
% outputDir = '';
% inDir = dir(fileDir);
% len = length(inDir);
% 
% for i=3:len
%     path_in = strcat(fileDir,inDir(i).name);
%     I = im2double(imread(path_in));
%     [m,n,c] = size(I);
%     % PNLS
%     tic,
%  
%     Idetexture = PNLS_DT(I);
%  
%     toc,
%     
%     out_PNLS = strcat(outputDir,'PNLS/',inDir(i).name);
%     imwrite(Idetexture, out_PNLS);
%     %print('ok');
% end


fileDir =  '/1T/datasets/VOC_SPS/val_small/' ;% input dir
outputDir = '/1T/WJ/Easy2Hard-master/test_results/VOC/';
inDir = dir(fileDir);
len = length(inDir);

parfor i=3:len
    path_in = strcat(fileDir,inDir(i).name);
    I = im2double(imread(path_in));
    [m,n,c] = size(I);
    I = I(:,n/2+1:n,:);
    % PNLS
    tic,
 
    Idetexture = PNLS_DT(I);
 
    toc,
    
    out_PNLS = strcat(outputDir,'PNLS/',inDir(i).name);
    imwrite(Idetexture, out_PNLS);
    %print('ok');
end