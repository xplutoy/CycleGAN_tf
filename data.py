import tensorflow as tf

from utils import convert2float


def datasets_info(name):
    return {
        'apple': ('datas/apple2orange/trainA', 'datas/apple.tfrecords'),
        'orange': ('datas/apple2orange/trainB', 'datas/orange.tfrecords'),
        'horse': ('datas/horse2zebra/trainA', 'datas/horse.tfrecords'),
        'zebra': ('datas/horse2zebra/trainB', 'datas/zebra.tfrecords'),
        'apple_t': ('datas/apple2orange/testA', 'datas/apple_t.tfrecords'),
        'orange_t': ('datas/apple2orange/testB', 'datas/orange_t.tfrecords'),
        'horse_t': ('datas/horse2zebra/testA', 'datas/horse_t.tfrecords'),
        'zebra_t': ('datas/horse2zebra/testB', 'datas/zebra_t.tfrecords')
    }[name]


class Reader:
    def __init__(self, name, image_size=256,
                 min_queue_examples=100, num_threads=4):
        self.name = name
        self.tfrecords_file = datasets_info(self.name)[1]
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.filename_queue = tf.train.string_input_producer([self.tfrecords_file])

    def feed(self, batch_size=1):
        """
        Returns:
            images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            _, serialized_example = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string)
                })

            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self._preprocess(image)

            images = tf.train.shuffle_batch(
                [image], batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples * 5, min_after_dequeue=self.min_queue_examples
            )
        return images

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image
