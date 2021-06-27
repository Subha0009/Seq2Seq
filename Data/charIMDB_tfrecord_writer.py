import sys
path_to_dir = sys.argv[1]
import pandas
import tensorflow as tf

df = pandas.read_csv('IMDB Dataset.csv')
df['label'] = df['sentiment']
df.loc[df['sentiment'] == 'negative', 'label'] = tf.cast(0, tf.int32)
df.loc[df['sentiment'] == 'positive', 'label'] = tf.cast(1, tf.int32)

for record_counter in range(0, 25000, 1000):
  with tf.io.TFRecordWriter(path_to_dir+'/train_{}.tfrecord'.format(record_counter)) as writer:
    for text, label in zip(df['review'][record_counter:record_counter+1000], df['label'][record_counter:record_counter+1000]):
      charlist = [ord(c) for c in text]
      example = tf.train.Example(features=tf.train.Features(feature={
      'input': tf.train.Feature(int64_list=tf.train.Int64List(value=charlist)),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
      writer.write(example.SerializeToString())
  print("Written {} out of {}".format(min(record_counter+1000, 25000),
                                      25000))

for record_counter in range(25000, 50000, 1000):
  with tf.io.TFRecordWriter(path_to_dir+'/test_{}.tfrecord'.format(record_counter)) as writer:
    for text, label in zip(df['review'][record_counter:record_counter+1000], df['label'][record_counter:record_counter+1000]):
      charlist = [ord(c) for c in text]
      example = tf.train.Example(features=tf.train.Features(feature={
      'input': tf.train.Feature(int64_list=tf.train.Int64List(value=charlist)),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
      writer.write(example.SerializeToString())
  print("Written {} out of {}".format(min(record_counter+1000, 50000)-25000,
                                      25000))
