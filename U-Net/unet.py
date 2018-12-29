import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from importdata import dataProcess
import cv2
import scipy.io


class UNet():

    def __init__(self, patch_rows, patch_cols, tr_img_rows, tr_img_cols, te_img_rows, te_img_cols):
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.tr_img_rows = tr_img_rows
        self.tr_img_cols = tr_img_cols
        self.te_img_rows = te_img_rows
        self.te_img_cols = te_img_cols

    def load_data_train(self):
        mydata = dataProcess(self.tr_img_rows, self.tr_img_cols)
        imgs_train, imgs_mask_train, imb_ratio = mydata.load_train_data()
        #imgs_test = mydata.load_test_data()
        #return imgs_train, imgs_mask_train, imgs_test
        return imgs_train, imgs_mask_train, imb_ratio

    def load_data_test(self):
        mydata = dataProcess(self.te_img_rows, self.te_img_cols)
        imgs_test = mydata.load_test_data()
        #imgs_test = mydata.load_test_data()
        #return imgs_train, imgs_mask_train, imgs_test
        return imgs_test

    def dice_coef(self, y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)=  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        y_true = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3))
        y_pred = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    '''def gen_dice_coef(self, y_true, y_pred, smooth=1e-7):
        
        #Dice coefficient for 10 categories. Ignores background pixel label 0
        #Pass to model as metric during compile statement
        
        #y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[..., 1:])
        #y_pred_f = K.flatten(y_pred[..., 1:])
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3))
        y_pred_f = K.flatten(y_pred)
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))

    def gen_dice_coef_loss(self, y_true, y_pred):
        
        #Dice loss to minimize. Pass to model as loss during compile statement
        
        return 1 - self.gen_dice_coef(y_true, y_pred)'''

    def patch_to_image(self, patch_pred):
        n_patches_x = int(self.te_img_cols / self.patch_cols)
        n_patches_y = int(self.te_img_rows/self.patch_rows)
        n_patches = n_patches_x * n_patches_y
        n_imgs = int(patch_pred.shape[0]/n_patches)
        im_pred = np.ndarray((n_imgs, self.te_img_rows, self.te_img_cols, 3), dtype=np.float32)
        for i in range(n_imgs):
            im_pred[i] = np.reshape(patch_pred[i*n_patches:(i*n_patches)+n_patches], (self.te_img_rows, self.te_img_cols,3), order='A')

        return im_pred

    '''def generalized_dice_coeff(y_true, y_pred):
        n_el = 1
        for dim in y_train.shape:
            n_el *= int(dim)
        n_cl = y_train.shape[-1]
        w = K.zeros(shape=(n_cl,))
        w = (K.sum(y_true, axis=(0, 1, 2))) / (n_el)
        w = 1 / (w ** 2 + 0.000001)
        numerator = y_true * y_pred
        numerator = w * K.sum(numerator, (0, 1, 2))
        numerator = K.sum(numerator)
        denominator = y_true + y_pred
        denominator = w * K.sum(denominator, (0, 1, 2))
        denominator = K.sum(denominator)
        return 2 * numerator / denominator''' ####byDAAN_KUPPENS

    def get_unet(self):
        inputs = Input((self. patch_rows, self.patch_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        print(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        print(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        print(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        print(conv9)
        #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        #print("conv9 shape:", conv9.shape)

        conv10 = Conv2D(3, 1, activation='softmax')(conv9)
        print(conv10)
        model = Model(input=inputs, output=conv10)

        #model.compile(optimizer=SGD(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=SGD(lr=1e-4, momentum=0.95), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #WORKS
        #model.compile(optimizer=SGD(lr=1e-5, momentum=0.95), loss=self.dice_coef_loss, metrics=[self.dice_coef]) #WORKS
        model.compile(optimizer=nadam(lr=1e-5), loss=self.dice_coef_loss, metrics=[self.dice_coef])

        return model

    def train(self):
        print("[INFO]: Loading training data...")
        #imgs_train, imgs_mask_train, imgs_test = self.load_data()
        imgs_train, imgs_mask_train, imb_ratio = self.load_data_train()
        print("[INFO]: Loading data done.")
        model = self.get_unet()
        print("[INFO]: UNet architecture loaded.")
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('[INFO]: Fitting model...')
        history = model.fit(imgs_train, imgs_mask_train, batch_size=64, epochs=80, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        # list all data in history
        print(history.history.keys())

        # Summarizing history for accuracy
        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('acc.png')

        # Summarizing history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('losses.png')
        plt.close('all')

        #print('predict test data')
        #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

        #np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def test(self):
        print("[INFO]: Loading test data...")
        imgs_test = self.load_data_test()
        print("[INFO]: UNet architecture loaded with the best trained weights.")
        model = self.get_unet()
        model.load_weights('./unet.hdf5')
        print("[INFO]: Predicting test data...")
        pred_test = model.predict(imgs_test, batch_size=64, verbose=1)
        #im_pred = self.patch_to_image(pred_test)
        #np.save('../results/pred_1024_test.npy', im_pred)
        #scipy.io.savemat('../results/pred_1024_test.mat', mdict={'pred_test': im_pred})
        #np.save('../results/imgs_mask_test.npy', pred_test)
        scipy.io.savemat('../results/pred_testDef_overlapping_dice80ep.mat', mdict={'pred_test': pred_test})
        #label_map = np.argmax(pred_test, axis=3)
        #pkk1 = prob_maps[1]

    def save_img(self):
        print("[INFO]: Array to image")
        imgs = np.load('../results/imgs_mask_test.npy')
        '''piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)'''
        for i in range(imgs.shape[0]):
            path = "../results/" + str(i) + ".png"
            img = imgs[i]
            label_map = np.argmax(img, axis=2)
            label_map[label_map == 1] = 128
            label_map[label_map == 2] = 255
            cv2.imwrite(path, label_map)
            #img = array_to_img(img)
            #img.save(path)
            #cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            #cv_pic = cv2.resize(cv_pic, (1918, 1280), interpolation=cv2.INTER_CUBIC)
            #binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)

if __name__ == '__main__':
    myunet = UNet(64, 64, 1000, 1000, 1024, 1024)
    # model = myunet.get_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    # myunet.train()
    myunet.test()
    #myunet.save_img()