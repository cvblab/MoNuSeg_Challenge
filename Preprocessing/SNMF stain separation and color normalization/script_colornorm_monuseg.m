% Main file to run different color normalization techniques
% For queries, contact: abhishek.vahadane@gmail.com, vahadane@iitg.ernet.in

clear
clc
close all

%% Input parameters
img_list = dir('../../../data/tissue_test_images/');
target = imread('../../../data/tissue_train_images/TCGA-E2-A14V-01Z-00-DX1.tif');
nstains=2;
lambda=0.02;  % Use smaller values of the lambda (0.01-0.1) for better reconstruction. however, if the normalized image seems not fine, increase or decrease the value accordingly.
verbose = 0;

%% Our Method (The techniques is published in ISBI 2015 under the title "STRUCTURE-PRESERVED COLOR NORMALIZATION FOR HISTOLOGICAL IMAGES")
for k = 3 : length(img_list)
    
    % Source
    source = imread(fullfile(img_list(k).folder,img_list(k).name));
    % Source and target stain separation and storage of factors
    tic
    [Wis, His,Hivs]=stainsep(source,nstains,lambda);
    % save('source.mat','Wis','His','Hivs')
    [Wi, Hi,Hiv]=stainsep(target,nstains,lambda);
    % save('target.mat','Wi','Hi','Hiv')
    
    % Color normlization
    % addpath(genpath('Our Method'))
    [our]=SCN(source,Hi,Wi,His);
    time=toc;
    
    % Write image to a folder
    imwrite(our,['../../../data/normalized_tissue_test_images/',img_list(k).name])
    %% Visuals
    if verbose
        figure;
        subplot(131);imshow(source);xlabel('source')
        subplot(132);imshow(target);xlabel('target')
        subplot(133);imshow(our);xlabel('normalized source')
    end
    
end