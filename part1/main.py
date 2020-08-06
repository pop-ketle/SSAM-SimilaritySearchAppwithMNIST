import os
import gzip
import pickle
import numpy as np
import urllib.request
from mnist import MNIST
from annoy import AnnoyIndex

from sklearn.metrics import accuracy_score

def load_mnist():
    # MNISTのデータをダウンロードしてくる
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    # ファイルをダウンロード(.gz形式)
    for filename in key_file.values():
        file_path = f'./static/mnist_data/{filename}' # 読み込んでくる.gzファイルのパス
        if os.path.isfile(file_path.replace('.gz', '')): continue # すでにファイルがあるときは飛ばす

        urllib.request.urlretrieve(url_base + filename, file_path)

        # .gzの解凍と.gzファイルの削除
        with gzip.open(file_path, mode='rb') as f:
            mnist = f.read()
            # 解凍して保存
            with open(file_path.replace('.gz', ''), 'wb') as w:
                w.write(mnist)
            os.remove(file_path) # .gzファイルの削除

    # mnistのデータをnp.arrayの形に読み込んで返す
    mndata = MNIST('./static/mnist_data/')
    # train
    images, labels = mndata.load_training()
    train_images, train_labels = np.reshape(np.array(images), (-1,28,28)), np.array(labels) # np.arrayに変換する mnistの画像は28x28
    # test
    images, labels = mndata.load_testing()
    test_images, test_labels = np.reshape(np.array(images), (-1,28,28)), np.array(labels) # np.arrayに変換する mnistの画像は28x28
    return train_images, train_labels, test_images, test_labels

def make_annoy_db(train_imgs):
    # 近似近傍探索のライブラリannoyを使う
    # 入力するデータのshapeとmetricを決めて、データを突っ込んでいく
    annoy_db = AnnoyIndex((28*28), metric='euclidean')
    for i, train_img in enumerate(train_imgs):
        annoy_db.add_item(i, train_img.flatten())
    annoy_db.build(n_trees=10) # ビルド
    annoy_db.save('./static/mnist_db.ann')


def main():
    # mnist画像の読み込み処理
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()
    print(train_imgs.shape, train_lbls.shape, test_imgs.shape, test_lbls.shape)

    if not os.path.isfile('./static/mnist_db.ann'):
        make_annoy_db(train_imgs) # annoydbのビルド
    annoy_db = AnnoyIndex((28*28), metric='euclidean')
    annoy_db.load('./static/mnist_db.ann') # annoyのデータベースをロードする


    # テストデータを入力して近い近傍を取ってきて実際と比較することで試しに精度をみてみる
    y_pred = [train_lbls[annoy_db.get_nns_by_vector(test_img.flatten(), 1)[0]] for test_img in test_imgs]
    score = accuracy_score(test_lbls, y_pred)
    print('acc:', score)



if __name__ == "__main__":
    main()