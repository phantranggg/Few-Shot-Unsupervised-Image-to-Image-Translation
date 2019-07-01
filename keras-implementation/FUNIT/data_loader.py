
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

def load_batch(batch_size=1):
    path = glob.glob('./datasets/102flowers/jpg/*')

    n_batches = int(len(path) / batch_size)
    total_samples = n_batches * batch_size

    path_content_imgs = np.random.choice(path, total_samples, replace=False)
    path_class_imgs = np.random.choice(path, 1, replace=False)

    k_samples = np.random.choice(path, k, replace=False)

    for i in range(n_batches - 1):
        batch_content_img = path_content_imgs[i * batch_size:(i + 1) * batch_size]
        batch_class_img = path_class_imgs[i * batch_size:(i + 1) * batch_size]
        content_imgs, class_imgs = [], []
        for content_img, class_img in zip(batch_content_img, batch_class_img):
            content_img = load_img(content_img)
            class_img = load_img(class_img)

            content_imgs.append(content_img)
            class_imgs.append(class_img)

        yield content_imgs, class_imgs