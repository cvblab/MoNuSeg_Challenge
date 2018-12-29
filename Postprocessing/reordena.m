function im=reordena(M)
im=zeros(1024,1024,3);
count=zeros(1024,1024);

for k=0:960
    col=rem(k,31);
    fil=floor(k/31);
    pfil=(fil+1)*32-32;
    if (fil==0) 
        pfil=0;
    end
    pcol=(col+1)*32-32;
    if (col==0)
        pcol=0;
    end
     %k
%      if (col==29) 
%          pcol=col+32 
%      end;
%      if (fil==29) 
%          pfil=fil+32 
%      end;
    im(pfil+1:pfil+64,pcol+1:pcol+64,:)=squeeze(M(k+1,:,:,:))+im(pfil+1:pfil+64,pcol+1:pcol+64,:);
    count(pfil+1:pfil+64,pcol+1:pcol+64)= count(pfil+1:pfil+64,pcol+1:pcol+64)+ones(64,64);
    
        
%  figure(1);imshow(mat2gray(im(:,:,2)));
%  figure(2);imshow(mat2gray(count(:,:)));
%  pause
end
im(:,:,1)=im(:,:,1)./count;
im(:,:,2)=im(:,:,2)./count;
im(:,:,3)=im(:,:,3)./count;

    