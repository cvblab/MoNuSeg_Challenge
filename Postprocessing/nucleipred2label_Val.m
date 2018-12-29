function [label_map, aji] = nucleipred2label_Val(im, pred_map, gt_map, thin, thext, w1)
% Model information
pred_map=imresize(pred_map,[1000 1000]);
fg=pred_map(:,:,2)>thin;
bg=pred_map(:,:,1)>thext;
%Colour deconvolution information
[H,~,~] = colour_deconvolution(im, 'H&E');
Hg=imdilate(H,strel('disk',1))-imerode(H,strel('disk',1));
% Watershed
iminOr = w1*uint8(255*mat2gray(pred_map(:,:,3)))+(1-w1)*uint8(255*mat2gray(Hg));
gradmark = imimposemin(iminOr, fg|bg);
imout = double(watershed(gradmark));
label_map = imdilate(imout,strel('disk',1));
label_map(label_map==1) = 0;
% AJI computation
%figure,imshow(label2rgb(label_map, 'jet', 'k'))
R = random_color(max(max(label_map))-1);
R(1,1) = 0; R(1,2)=0; R(1,3)=0;
imwrite(ind2rgb(label_map, R),'imagen1.png');
aji = aggregated_jaccard_index(gt_map, label_map);
