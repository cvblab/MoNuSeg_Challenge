% to generate markers, binary map and coloured maps from probability maps
clc;
%clear all;

% boundary = imread('1.png');%give boundary map as input
% boundary = boundary(:,:,1);
% 
% nucleus = imread('2.png');%give nuclei maps as input
boundary = uint8(im(:,:,3)*255);
nucleus = uint8(im(:,:,2)*255);
nucleus = im2bw(nucleus,.4); %threshold may be changed for different probability maps
imwrite(nucleus, 'marker.png'); %gives markers for the nuclei

conComp = bwlabel(nucleus);
dim = size(nucleus);
image_bw = zeros(dim(1),dim(2)); % binary map
image_col = zeros(dim(1),dim(2),3); % colored object map
se = [0 1 0; 1 1 1; 0 1 0];
current = zeros(1,max(conComp(:)));
bound_mean = mean(boundary(boundary ~= 0));


for i = 1 : max(conComp(:))
    temp_bw = zeros(dim(1), dim(2));
    temp_bw(conComp == i) = 1;
    temp_col = zeros(dim(1), dim(2),3);
    temp_col(:,:,1) = rand*temp_bw;
    temp_col(:,:,2) = rand*temp_bw;
    temp_col(:,:,3) = rand*temp_bw;
    prev = 0;
    count = 0;
    while(current(1,i) < bound_mean && count < 2)
        prev = current(1,i);
        temp1 = imdilate(temp_bw, se);
        temp2 = imdilate(temp_col, se);
        count = count + 1;
        temp_diff =uint8(temp1 - temp_bw);
        temp_mult = (temp_diff) .* boundary;
        %figure(), imshow(temp_mult);
        current(1,i) = mean(temp_mult(temp_mult ~= 0));
        disp(current(1,i))
         %if(current(1,i)>=prev)
            temp_bw = temp1;
            temp_col = temp2;
         %end
    end
    image_bw = image_bw + temp_bw;
    image_col = image_col + temp_col;
end
figure(1)
imshow(image_bw);
figure(2)
imshow(image_col);
imwrite(image_bw, 'bin_map.png');
imwrite(image_col, 'col_map.png');