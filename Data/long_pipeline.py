import tensorflow as tf
import tensorflow_datasets as tfds

def _load_records(filename):
  return tf.data.TFRecordDataset(filename, buffer_size=8 * 1000 * 1000)


def _parse_example(serialized_example):
  data_fields = {
      "input": tf.io.VarLenFeature(tf.int64),
      "label": tf.io.VarLenFeature(tf.int64)
  }
  parsed = tf.io.parse_single_example(serialized_example, data_fields)
  input = tf.sparse.to_dense(parsed["input"])
  label = tf.sparse.to_dense(parsed["label"])
  return tf.cast(input, dtype=tf.int32), tf.cast(label, dtype=tf.int32)

def _get_example_length(i1, i2):
    return tf.shape(i1)[0]

def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.size(example[0]) <= max_length

def _create_min_max_boundaries(max_length):
    bucket_boundaries = []
    x = 50
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * 1.1))
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset, boundary_window, batch_size=16000, max_length=14000):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(i1, i2):
        seq_length = _get_example_length(i1, i2)

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
                                            padding_values=(tf.constant(0, dtype=tf.int32),
                                                            None))

    return dataset.apply(
                      tf.data.experimental.group_by_window(
                      key_func=example_to_bucket_id,
                      reduce_func=batching_fn,
                      window_size=None,
                      window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def make_dataset(data_dir, 
                 batch_size=64000, 
                 maxlen=4000, 
                 split='train',
                 num_replicas=8,
                 static_batch=True):
    dataset = tf.data.Dataset.list_files(data_dir+'/{}*'.format(split), shuffle=True)
    dataset = dataset.interleave(_load_records,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), maxlen))
    if not static_batch:
        return _batch_examples(dataset,
                           boundary_window,
                           batch_size=batch_size, 
                           max_length=maxlen)
    else:
        return dataset.padded_batch(int(batch_size // num_replicas // maxlen * num_replicas), 
                                    padded_shapes=([maxlen],[]),
                                    padding_values=(tf.constant(0, dtype=tf.int32), None),
                                    drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
