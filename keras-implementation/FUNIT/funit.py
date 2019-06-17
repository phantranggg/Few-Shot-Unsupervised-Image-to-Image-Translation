import numpy as np
import tensorflow as tf
from keras.layers import Input, Add, Conv2D, BatchNormalization, ReLU, LeakyReLU, UpSampling2D, Concatenate, AveragePooling2D, Activation, ZeroPadding2D, MaxPooling2D, Dense, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import imageio
from PIL import Image
from matplotlib.pyplot import imshow
import datetime
import random
import scipy.io
import glob
%matplotlib inline

import keras.backend as K

class FUNIT():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def content_encoder(shape):
            X_input = Input(shape=shape)

            X = Conv2D(64, kernel_size=1, strides=1, padding='same')(X_input)
            X = InstanceNormalization()(X)
            X = ReLU()(X)

            X = Conv2D(128, kernel_size=1, strides=2, padding='same')(X)
            X = InstanceNormalization()(X)
            X = ReLU()(X)

            X = Conv2D(256, kernel_size=1, strides=2, padding='same')(X)
            X = InstanceNormalization()(X)
            X = ReLU()(X)

            X = Conv2D(512, kernel_size=1, strides=2, padding='same')(X)
            X = InstanceNormalization()(X)
            X = ReLU()(X)

            X_shortcut = X

            X = Conv2D(512, kernel_size=1, strides=1, padding='same')(X)
            X = InstanceNormalization()(X)
            X = Add()([X_shortcut, X])
            X = ReLU()(X)

            X_shortcut = X

            X = Conv2D(512, kernel_size=1, strides=1, padding='same')(X)
            X = InstanceNormalization()(X)
            X = Add()([X_shortcut, X])
            X = ReLU()(X)

            #     model = Model(inputs=X_input, outputs=X, name='content_encoder')
            #     return model

            return X

        def class_encoder_k(shape):
            Y_input = Input(shape=shape)

            Y = Conv2D(64, kernel_size=1, strides=1, padding='same')(Y_input)
            Y = ReLU()(Y)

            Y = Conv2D(128, kernel_size=1, strides=2, padding='same')(Y)
            Y = ReLU()(Y)

            Y = Conv2D(256, kernel_size=1, strides=2, padding='same')(Y)
            Y = ReLU()(Y)

            Y = Conv2D(512, kernel_size=1, strides=2, padding='same')(Y)
            Y = ReLU()(Y)

            Y = Conv2D(1024, kernel_size=1, strides=2, padding='same')(Y)
            Y = ReLU()(Y)

            Y = AveragePooling2D()(Y)

            #     Y = Dense(512)(Y)
            Y = Flatten()(Y)

            return Y

        def class_encoder(Y):
            total = 0
            for i in range(0, k):
                val = class_encoder_k(Y.shape)
                total = np.add(total, val)
            mean = tf.math.reduce_mean(total)
            print(mean)
            return mean

        def adain_resblk(X, f, filters, stage, s):
            conv_name_base = 'adain_resblk_' + stage
            F1, F2, F3 = filters

            X_shortcut = X
            print(X_shortcut)

            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = InstanceNormalization(beta_regularizer=beta_2, gamma_regularizer=gamma)(X)
            X = ReLU()(X)
            print(X)

            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = InstanceNormalization(beta_regularizer=beta_2, gamma_regularizer=gamma)(X)
            X = ReLU()(X)
            print(X)

            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                       kernel_initializer=glorot_uniform(seed=0))(X)

            X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

            print(X, X_shortcut)
            X = Add()([X, X_shortcut])
            X = ReLU()(X)

            return X

        def conv(X, f_size):
            X = Conv2D(X, kernel_size=f_size, strides=2, padding='valid')(X)
            X = InstanceNormalization()(X)
            X = ReLU()(X)
            return X

        def decoder(content_code, class_code):
            X = Dense(256)(class_code)
            X = Dense(256)(X)
            X = Dense(256)(X)
            X = Flatten()(X)

            beta = np.var(X, axis=(1, 2))
            beta_2 = beta ** 2
            gamma = np.mean(X, axis=(1, 2))

            Y = adain_resblk(content_code, f=3, filters=[512, 512, 512], stage=1, s=1)
            Y = adain_resblk(Y, f=3, filters=[512, 512, 512], stage=2, s=1)
            Y = conv(Y, f_size=256)
            Y = conv(Y, f_size=128)
            Y = conv(Y, f_size=64)
            Y = conv(Y, f_size=3)

            return Y

        # Image input
        X = Input(shape=self.img_shape)
        # Y = Input(shape=self.img_shape)

        content_code = content_encoder(X_input.shape)
        class_code = class_encoder(Y)
        X_gen = decoder(content_code, class_code)
        model = Model(inputs=[X_input, Y], outputs=X_gen)
        return model

    def build_discriminator(self):

        def res_block(X, f, filters, stage, s):
            conv_name_base = 'resblk_' + str(stage)
            F1, F2, F3 = filters

            X_shortcut = X
            print(X_shortcut)

            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + 'a',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = LeakyReLU()(X)
            print(X)

            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + 'b',
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = LeakyReLU()(X)
            print(X)

            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + 'c',
                       kernel_initializer=glorot_uniform(seed=0))(X)

            X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

            print(X, X_shortcut)
            X = Add()([X, X_shortcut])
            X = LeakyReLU()(X)

            return X

        X_input = Input(img_shape)

        X = Conv2D(64, kernel_size=3, strides=2, padding='same')(X_input)

        X = res_block(X, f=3, filters=[64, 64, 128], stage=1, s=1)
        X = res_block(X, f=3, filters=[128, 128, 128], stage=2, s=1)

        X = AveragePooling2D(strides=2)(X)

        X = res_block(X, f=3, filters=[128, 128, 256], stage=3, s=1)
        X = res_block(X, f=3, filters=[256, 256, 256], stage=4, s=1)

        X = AveragePooling2D(strides=2)(X)

        X = res_block(X, f=3, filters=[256, 256, 512], stage=6, s=1)
        X = res_block(X, f=3, filters=[512, 512, 512], stage=7, s=1)

        X = AveragePooling2D(strides=2)(X)

        X = res_block(X, f=3, filters=[512, 512, 1024], stage=8, s=1)
        X = res_block(X, f=3, filters=[1024, 1024, 1024], stage=9, s=1)

        X = AveragePooling2D(strides=2)(X)

        X = res_block(X, f=3, filters=[1024, 1024, 1024], stage=11, s=1)
        X = res_block(X, f=3, filters=[1024, 1024, 1024], stage=12, s=1)

        number_source_class = 10
        X = Conv2D(number_source_class, kernel_size=1, strides=1, padding='same')(X)

        model = Model(inputs=X_input, outputs=X)
        return model

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=100, batch_size=1, sample_interval=20)