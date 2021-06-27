import tensorflow as tf
import pickle

def line_reader(inputList, refList):
  def line_generator():
    for en_line, de_line in zip(inputList, refList):
      yield tf.convert_to_tensor(en_line, dtype=tf.int32), tf.convert_to_tensor(de_line, dtype=tf.string)
  return tf.data.Dataset.from_generator(line_generator, (tf.int32, tf.string), (tf.TensorShape([None,]), tf.TensorShape([])))

def _get_example_length(i):
    return tf.shape(i)[0]

def _create_min_max_boundaries(max_length=200,
                               min_boundary=3,
                               boundary_scale=1.1):
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length, min_boundary):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length, min_boundary=min_boundary)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(i1, i2):
        seq_length = _get_example_length(i1)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size, 
                                            padded_shapes=([None],[]),
                                            padding_values=(0, None))

    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def make_dataset(tokenizer, src_file, tar_file, batch_size=4000, maxlen=200, min_boundary=2):
    with open(src_file, 'rb') as f:
        test_src = pickle.load(f)
    with open(tar_file, 'rb') as f:
        test_tar = pickle.load(f)
    test_tar = [tokenizer.decode(line[1:-1]) for line in test_tar]
    return _batch_examples(line_reader(test_src, test_tar),
                           batch_size,
                           maxlen,
                           min_boundary)
