img_name = 'TCGA-NH-A8F7-01A-01-TS1';
%img_name = 'TCGA-RD-A8N9-01A-01-TS1';
im = imread(['./normalized_train_images/',img_name,'.tif']);
load(['./Valery_validation/',img_name,'_gt_map.mat'])
%load(['./Valery_validation/',img_name,'_predicted_map.mat'])
load(['pred_test_overlapping_dice80ep.mat'])
%predicted_map = reordena(pred_test(962:end,:,:,:));%AJI = 0.7025; Dice=0.8823 // AJI=0.7273 ; Dice= 0.8856
predicted_map = reordena(pred_test(1:961,:,:,:)); %AJI=0.5293 ; Dice=0.7615 // AJI=0.5195 ; Dice=0.7729
thint = 0.75;
thext = 0.75;
w1 = 0.3;
%thint = 0.95;
%thext = 0.8;
%w1 = 0.2;

[label_map, aji] = nucleipred2label_Val(im, predicted_map, gt_map, thint, thext, w1);
