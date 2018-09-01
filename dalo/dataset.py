from dalo.pascal import Pascal
from tqdm import tqdm
import os
import tensorflow as tf


def build_dataset():
    def mapper(input):
        image = tf.read_file(input['image_file'])
        image = tf.image.decode_png(image, channels=3)

        return {
            'image': image,
            'class_ids': input['class_ids'],
            'boxes': input['boxes']
        }

    dl = Pascal(os.path.expanduser('~/Datasets/pascal/VOCdevkit/VOC2012'), 'trainval')
    ds = (tf.data.Dataset.from_generator(
        lambda: dl,
        output_types={'image_file': tf.string, 'class_ids': tf.int32, 'boxes': tf.float32},
        output_shapes={'image_file': [], 'class_ids': [None], 'boxes': [None, 4]})
          .map(mapper, num_parallel_calls=os.cpu_count())
          .prefetch(None))

    return ds


def main():
    ds = build_dataset()
    iter = ds.make_one_shot_iterator()
    input = iter.get_next()

    with tf.Session() as sess:
        for _ in tqdm(range(10000)):
            x = sess.run(input)
            assert x['image'].shape[2] == 3


if __name__ == '__main__':
    main()
