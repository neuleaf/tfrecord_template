import tensorflow as tf
from PIL import Image
import numpy as np


def read_tfrecord(filename_queue):
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/height': tf.FixedLenFeature([], tf.int64),
               'image/width': tf.FixedLenFeature([], tf.int64),
               'image/label': tf.FixedLenFeature([], tf.int64)}

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image  = tf.decode_raw(features['image/encoded'], tf.uint8)
    height = tf.cast(features['image/height'],tf.int32)
    width  = tf.cast(features['image/width'], tf.int32)
    label  = tf.cast(features['image/label'], tf.int32)

    # reshape,恢复成图像的shape；这里height，widht，ch本可以使用上边解析出来的height，width
    # 但是如果后续使用batch获取数据的话则必须使用静态的常量来指定shape
    # 但是，我们可以在preprocess中，再具体设定shape
    img = tf.reshape(image, [height, width, 3])

    # optional preprocess
    # rand crop
    # tf.random_crop()
    # img = tf.image.resize_images(img, [224,224])
    # img = tf.cast(img, tf.float32) * (2. / 255) - 1.0

    return img, label


def get_batch(infile, batch_size, num_threads=4, shuffle=False, min_after_dequeue=None):
    # 使用batch，img的shape必须是静态常量
    image, label = read_tfrecord(infile)

    if min_after_dequeue is None:
        min_after_dequeue = batch_size * 10
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                    capacity=capacity,num_threads=num_threads,
                                                    min_after_dequeue=min_after_dequeue)
    else:
        img_batch, label_batch = tf.train.batch([image, label], batch_size,
                                                capacity=capacity, num_threads=num_threads,
                                                allow_smaller_final_batch=True)

    return img_batch, label_batch


if __name__=='__main__':
    import matplotlib.pyplot as plt

    # make file queue.
    # 参数'num_epochs'如果指定数量，结合训练时循环条件cood.should_stop()，
    # 可以起到控制训练迭代次数的作用。
    # 如果为None，队列会一直循环读tfrecord中的数据；直到退出训练迭代
    filename_queue = tf.train.string_input_producer(['./tfrecords/imagenet.tfrecord'],
                                                    num_epochs=None)
    img, label = read_tfrecord(filename_queue)

    # or, use batch
    # img_batch, label_batch= get_batch(filename_queue, batch_size=4)

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2):
            if coord.should_stop():
                break

            imgs, ls = sess.run([img, label])
            print(imgs.shape)
            plt.imshow(imgs)
            plt.show()

            # or, use batch
            # imgs, ls = sess.run([img_batch, label_batch])
            # print(ls)
            # print(imgs.shape)

        coord.request_stop()
        coord.join(threads)