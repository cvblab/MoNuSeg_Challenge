from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class dataProcess(object):
    def __init__(self, out_rows, out_cols, patch_size=64, n_classes=3, samples_per_class=4500, data_path="../data/new_partition/normalized_tissue_images/train/",
                 label_path="../data/new_partition/annotations/train/", test_path="../data/new_partition/normalized_tissue_images/test/",
                 npy_path="../data/npydata_new/", img_type="tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.samples_per_class = samples_per_class
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def pixel_class_per_image(self, label):
        # Finding pixel positions depending annotations
        bck_pix = np.where(label == 0)
        nuc_pix = np.where(label == 1)
        bound_pix = np.where(label == 2)

        return bck_pix[0].shape[0], nuc_pix[0].shape[0], bound_pix[0].shape[0]

    def random_patch_sampling(self, img, label):
        # Normalizing image
        img /= 255.
        # Finding pixel positions depending annotations
        idx_bck = np.where(label == 0)
        idx_nuc = np.where(label == 1)
        idx_bound = np.where(label == 2)
        # Removing out-of-bounds pixels according to the patch size
        bck_remove = np.concatenate((np.where(idx_bck[0] < self.patch_size), np.where(idx_bck[1] < self.patch_size),
            np.where(idx_bck[0] > self.out_rows-self.patch_size), np.where(idx_bck[1] > self.out_cols-self.patch_size)), axis=1)
        nuc_remove = np.concatenate((np.where(idx_nuc[0] < self.patch_size), np.where(idx_nuc[1] < self.patch_size),
            np.where(idx_nuc[0] > self.out_rows-self.patch_size), np.where(idx_nuc[1] > self.out_cols-self.patch_size)), axis=1)
        bound_remove = np.concatenate((np.where(idx_bound[0] < self.patch_size), np.where(idx_bound[1] < self.patch_size),
            np.where(idx_bound[0] > self.out_rows-self.patch_size), np.where(idx_bound[1] > self.out_cols-self.patch_size)), axis=1)
        x_bck = np.delete(idx_bck[0], bck_remove)
        y_bck = np.delete(idx_bck[1], bck_remove)
        x_nuc = np.delete(idx_nuc[0], nuc_remove)
        y_nuc = np.delete(idx_nuc[1], nuc_remove)
        x_bound = np.delete(idx_bound[0], bound_remove)
        y_bound = np.delete(idx_bound[1], bound_remove)
        # Random selection of #samples_per_class patches
        idx_rand_bck = np.random.RandomState(seed=42).permutation(np.arange(x_bck.shape[0]))[0:(self.samples_per_class)]
        idx_rand_nuc = np.random.RandomState(seed=42).permutation(np.arange(x_nuc.shape[0]))[0:(self.samples_per_class)]
        idx_rand_bound = np.random.RandomState(seed=42).permutation(np.arange(x_bound.shape[0]))[0:(self.samples_per_class)]
        #Inicialization
        patch_img = np.ndarray((self.samples_per_class*self.n_classes, self.patch_size, self.patch_size, 3), dtype=np.float32)
        patch_mask = np.ndarray((self.samples_per_class*self.n_classes, self.patch_size, self.patch_size, 1), dtype=np.uint8)
        # Cropping patches
        p = 0
        for k in range(0, len(idx_rand_bck)):
            patch_img[p] = img[int(x_bck[idx_rand_bck[k]]- np.floor(self.patch_size/2)) : int(x_bck[idx_rand_bck[k]]+ np.floor(self.patch_size/2)),
                        int(y_bck[idx_rand_bck[k]] - np.floor(self.patch_size / 2)): int(y_bck[idx_rand_bck[k]] + np.floor(self.patch_size / 2)), :]
            #patch_img[p] = patch_img[p] /float(np.max(patch_img[p]))
            #patch2write = patch_img[p]*255
            #cv2.imwrite('../data/normalized_tissue_images/train/patches/patch_'+str(x_bck[idx_rand_bck[k]])+'_'+str(y_bck[idx_rand_bck[k]])+'.png', patch2write.astype(int))
            patch_mask[p] = label[int(x_bck[idx_rand_bck[k]] - np.floor(self.patch_size / 2)): int(x_bck[idx_rand_bck[k]] + np.floor(self.patch_size / 2)),
                        int(y_bck[idx_rand_bck[k]] - np.floor(self.patch_size / 2)): int(y_bck[idx_rand_bck[k]] + np.floor(self.patch_size / 2)), :]
            #mask2write = patch_mask[p]
            #mask2write[mask2write == 1] = 128
            #mask2write[mask2write == 2] = 255
            #cv2.imwrite('../data/annotations/train/patches/patch_' + str(x_bck[idx_rand_bck[k]]) + '_' + str(y_bck[idx_rand_bck[k]])+'.png', mask2write)
            p += 1
        for k in range(0, len(idx_rand_nuc)):
            patch_img[p] = img[int(x_nuc[idx_rand_nuc[k]]-np.floor(self.patch_size/2)) : int(x_nuc[idx_rand_nuc[k]]+np.floor(self.patch_size/2)),
                        int(y_nuc[idx_rand_nuc[k]] - np.floor(self.patch_size / 2)): int(y_nuc[idx_rand_nuc[k]] + np.floor(self.patch_size / 2)), :]
            #patch_img[p] = patch_img[p] / float(np.max(patch_img[p]))
            patch_mask[p] = label[int(x_nuc[idx_rand_nuc[k]] - np.floor(self.patch_size / 2)): int(x_nuc[idx_rand_nuc[k]] + np.floor(self.patch_size / 2)),
                        int(y_nuc[idx_rand_nuc[k]] - np.floor(self.patch_size / 2)): int(y_nuc[idx_rand_nuc[k]] + np.floor(self.patch_size / 2)), :]
            p += 1
        for k in range(0, len(idx_rand_bound)):
            patch_img[p] = img[int(x_bound[idx_rand_bound[k]]-np.floor(self.patch_size/2)) : int(x_bound[idx_rand_bound[k]]+np.floor(self.patch_size/2)),
                        int(y_bound[idx_rand_bound[k]] - np.floor(self.patch_size / 2)): int(y_bound[idx_rand_bound[k]] + np.floor(self.patch_size / 2)), :]
            #patch_img[p] = patch_img[p] / float(np.max(patch_img[p]))
            patch_mask[p] = label[int(x_bound[idx_rand_bound[k]] - np.floor(self.patch_size / 2)): int(x_bound[idx_rand_bound[k]] + np.floor(self.patch_size / 2)),
                        int(y_bound[idx_rand_bound[k]] - np.floor(self.patch_size / 2)): int(y_bound[idx_rand_bound[k]] + np.floor(self.patch_size / 2)), :]
            p += 1

        return patch_img, patch_mask

    def slice_test_data(self, img):
        # Normalizing image
        img /= 255.
        # Inicialization
        patch_img = []
        #for x in range(0, img.shape[0]-self.patch_size, self.patch_size):
        #    for y in range(0, img.shape[1]-self.patch_size, self.patch_size):
        for x in range(0, img.shape[0]-int(self.patch_size/2), int(self.patch_size/2)):
            for y in range(0, img.shape[1]-int(self.patch_size/2), int(self.patch_size/2)):
                my_te_patch = img[x:x+self.patch_size, y:y+self.patch_size, :]
                patch_img.append(my_te_patch)
        return np.asarray(patch_img, dtype=np.float32)

    def create_train_data(self):
        i = bck_count = nuc_count = bound_count = 0
        print('[INFO]: Creating training images...')
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        imgdatas = np.ndarray((len(imgs)*self.samples_per_class*self.n_classes, self.patch_size, self.patch_size, 3), dtype=np.float32)
        imglabels = np.ndarray((len(imgs)*self.samples_per_class*self.n_classes, self.patch_size, self.patch_size, 1), dtype=np.float32)
        #imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        #imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for x in range(len(imgs)):
            imgpath = imgs[x]
            pic_name = imgpath.split('/')[-1]
            labelpath = self.label_path + pic_name[:-3] + 'bmp'
            img = load_img(imgpath, grayscale=False, target_size=[self.out_rows, self.out_cols])
            label = load_img(labelpath, grayscale=True, target_size=[self.out_rows, self.out_cols])
            img = img_to_array(img)
            label = img_to_array(label)
            bck_pix, nuc_pix, bound_pix = self.pixel_class_per_image(label) # Class imbalance info
            img_patch, label_patch = self.random_patch_sampling(img,label)
            imgdatas[(i*self.samples_per_class*self.n_classes):((i+1)*self.samples_per_class*self.n_classes)] = img_patch
            imglabels[(i*self.samples_per_class*self.n_classes):((i+1)*self.samples_per_class*self.n_classes)] = label_patch
            print('[INFO]: Random patches for image {0}/{1} were successfully extracted.'.format(i+1, len(imgs)))
            bck_count += bck_pix
            nuc_count += nuc_pix
            bound_count += bound_pix
            i += 1

        pix_total = self.out_rows*self.out_cols*len(imgs)
        mdict = {'0': (pix_total-bck_count)/pix_total, '1': (pix_total-nuc_count)/pix_total, '2': (pix_total-bound_count)/pix_total}
        print('[INFO]: Patch extraction completed.')
        np.save(self.npy_path + 'imgs_train.npy', imgdatas)
        np.save(self.npy_path + 'imgs_mask_train.npy', imglabels)
        np.save(self.npy_path + 'imbalanced_ratio.npy', mdict)
        print('[INFO]: Saving to imgs_train.npy files done.')

    def create_test_data(self):
        print('[ÌNFO]: Creating test slices...')
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        #imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []
        patch_te_list = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[self.out_rows, self.out_cols])
            img = img_to_array(img)
            te_patches = self.slice_test_data(img)
            patch_te_list.append(te_patches)
        imgdatas = np.vstack(patch_te_list)
        txtname = '../results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('[INFO]: Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('[INFO]: Loading train images...')
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imb_ratio = np.load(self.npy_path + "/imbalanced_ratio.npy")
        #imgs_train = imgs_train.astype('float32')
        #imgs_mask_train = imgs_mask_train.astype('float32')
        #imgs_train /= 255
        #imgs_mask_train /= 255
        #imgs_mask_train[imgs_mask_train > 0.5] = 1  # 白
        #imgs_mask_train[imgs_mask_train <= 0.5] = 0  # 黑
        return imgs_train, imgs_mask_train, imb_ratio

    def load_test_data(self):
        print('-' * 30)
        print('[INFO]: Loading test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        #imgs_test = imgs_test.astype('float32')
        #imgs_test /= 255
        return imgs_test



if __name__ == "__main__":
    #Training
    #mydata_tr = dataProcess(1000, 1000)
    #mydata_tr.create_train_data()
    mydata_te = dataProcess(1024, 1024)
    mydata_te.create_test_data()