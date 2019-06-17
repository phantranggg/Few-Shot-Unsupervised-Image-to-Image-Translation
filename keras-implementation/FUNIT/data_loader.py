
def load_img(path):
    img = Image.open(path)
    img = img.resize((img_rows, img_cols), Image.ANTIALIAS)
    img = np.asarray(img)/127.5 - 1.
    return img[np.newaxis, :, :, :]

def load_data(batch_size=1):
    path = glob.glob('./datasets/102flowers/jpg/*')

    batch_images = random.sample(path, batch_size)

    imgs = []
    for img_path in batch_images:
        img = Image.open(img_path)
        img = img.resize((img_rows, img_cols), Image.ANTIALIAS)
        img.save(img_path)
        img = np.array(img)
        imgs.append(img)

    imgs = np.array(imgs)/127.5 - 1.

    return imgs

def load_batch(batch_size=1, k=1):
    path = glob.glob('./datasets/102flowers/jpg/*')

    n_batches = int(len(path) / batch_size)
    total_samples = n_batches * batch_size

    path_content_imgs = np.random.choice(path, total_samples, replace=False)
    path_class_img = np.random.choice(path, 1, replace=False)
    for i in range(len(path)):
#         if (train[i] == )
    k_samples = np.random.choice(path, k, replace=False)

    for i in range(n_batches-1):
        batch_content_img = samples[i*batch_size:(i+1)*batch_size]
        batch_class_img = [range(batch_size), k_samples]
        imgs_A, imgs_B = [], []
        for img_A, img_B in zip(batch_A, batch_B):
            img_A = self.imread(img_A)
            img_B = self.imread(img_B)

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        yield imgs_A, imgs_B