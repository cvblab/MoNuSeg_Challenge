
img_name = 'TCGA-NH-A8F7-01A-01-TS1';
%img_name = 'TCGA-RD-A8N9-01A-01-TS1';
im = imread(['./normalized_train_images/',img_name,'.tif']);
load(['./Valery_validation/',img_name,'_gt_map.mat'])
%load(['./Valery_validation/',img_name,'_predicted_map.mat'])
load(['pred_test_overlapping_dice80ep.mat'])
%predicted_map = reordena(pred_test(962:end,:,:,:));
predicted_map = reordena(pred_test(1:961,:,:,:));

v_thint = 0.5 : 0.05 : 1;
v_thext = 0.5 : 0.05 : 1;
v_w1 = 0 : 0.05 : 0.5;

combCoarse = combvec(v_thint, v_thext, v_w1);

parfor k = 1 : size(combCoarse,2)
    thint = combCoarse(1,k);
    thext = combCoarse(2,k);
    w1 = combCoarse(3,k);
    [label_map, aji] = nucleipred2label_Val(im, predicted_map, gt_map, thint, thext, w1);
    outOpt{k} = [thint thext w1 aji];
     disp(['[INFO]: Threshold internal marker = ',num2str(thint),...
        ' & threshold external marker = ',num2str(thext), ' & w1 = ', num2str(w1), ' & AJI = ', num2str(aji)]);
end
