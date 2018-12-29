function [nuclei, aji] = nucleipred2label(im, gt, thin, thext)

im = imresize(im, [1000 1000]);
mrkin = im(:,:,2)>thin;
mrkext = im(:,:,1)>thext;
gradmark = imimposemin(im(:,:,3), mrkin | mrkext);
L = double(watershed(gradmark));
L=L-1;
%L(L==1)=0;
nuclei = imclose(L, strel('disk',1));
figure,imshow(label2rgb(nuclei, 'jet', 'k'))
aji = aggregated_jaccard_index(gt, nuclei);


