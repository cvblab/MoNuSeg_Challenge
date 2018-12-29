
% Input parameters
thint = 0.7;
thext = 0.7;
w1 = 0.25;
test_path = './normalized_test_images/';

%Loading predictions and reading test path
load('pred_testDef_overlapping_dice80ep.mat')
test_files = textread('pic.txt','%s');

%Inicialization
idx = 1;
t = 0;
for k = 1 : size(test_files,1)
    img_name = strsplit(test_files{k},'/');
    img_name = img_name{end};
    im = imread(fullfile(test_path, img_name));
    disp(['[INFO]: Segmenting nuclei for image ',img_name]);
    data = pred_test(1:961,:,:,:);
    tic
    predicted_map = reordena(data);
    label_map = nucleipred2label_test(img_name, im, predicted_map, thint, thext, w1);
    t = t + toc;
    pred_test(1:961,:,:,:) = [];
end
time = t/k