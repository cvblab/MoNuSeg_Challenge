function label_map = nucleipred2label_test(img_name, im, pred_map, thin, thext, w1)
% Path to save the results
res_path = ['./resultsMoNuSeg/thin',num2str(thin),'_thext',num2str(thext),'_w',num2str(w1)];
% Check folder
if ~exist(res_path, 'dir')
    mkdir(res_path);
end
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
R = random_color(max(max(label_map))-1);
R(1,1) = 0; R(1,2)=0; R(1,3)=0;
%figure,imshow(ind2rgb(label_map, R))
%figure,imshow(label2rgb(label_map, 'jet', 'k'))
%figure,imshow(imoverlay(im, label_map>0))
imwrite(ind2rgb(label_map, R),[res_path,'/',img_name(1:end-4),'_label_map.png']);
imwrite(imoverlay(im, label_map>0),[res_path,'/',img_name(1:end-4),'_overlay.png']);
imwrite(im,[res_path,'/',img_name(1:end-4),'.png']);
save([res_path,'/',img_name(1:end-4),'.mat'],'label_map');
%aji = aggregated_jaccard_index(gt_map, label_map);