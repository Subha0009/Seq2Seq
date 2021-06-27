import sys
path_to_dir = sys.argv[1]
import tensorflow as tf
import json
import glob

vocab = {}
counter = 1
sym_index = 1
lines = []
for fname in glob.glob('spinn/python/spinn/data/listops/train_*'):
  with open(fname) as f:
    line = f.readlines()
    cline = [l for l in lines if len(l.split('\t')[1].split())>=500 and len(l.split('\t')[1].split())<=2000]
    lines.extend(cline)

for record_counter in range(0, len(lines), 4000):
  with tf.io.TFRecordWriter(path_to_dir+'/train_{}.tfrecord'.format(record_counter)) as writer:
    for line in lines[record_counter:record_counter+4000]:
      label, exp = line.split('\t')
      sym_list = []
      for sym in exp.split():
        if sym not in vocab.keys():
          vocab[sym] = sym_index
          sym_index += 1
        sym_list.append(vocab[sym])
      example = tf.train.Example(features=tf.train.Features(feature={
          'input': tf.train.Feature(int64_list=tf.train.Int64List(value=sym_list)),
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))}))
      writer.write(example.SerializeToString())
  print("Written {} out of {}".format(min(record_counter+4000, len(lines)),
                                      len(lines)))

lines = []
for fname in glob.glob('spinn/python/spinn/data/listops/test_*'):
  with open(fname) as f:
    line = f.readlines()
    cline = [l for l in lines if len(l.split('\t')[1].split())>=500 and len(l.split('\t')[1].split())<=2000]
    lines.extend(cline)

for record_counter in range(0, len(lines), 4000):
  with tf.io.TFRecordWriter(path_to_dir+'/test_{}.tfrecord'.format(record_counter)) as writer:
    for line in lines[record_counter:record_counter+4000]:
      label, exp = line.split('\t')
      sym_list = []
      for sym in exp.split():
        sym_list.append(vocab[sym])
      example = tf.train.Example(features=tf.train.Features(feature={
          'input': tf.train.Feature(int64_list=tf.train.Int64List(value=sym_list)),
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))}))
      writer.write(example.SerializeToString())
  print("Written {} out of {}".format(min(record_counter+4000, len(lines)),
                                      len(lines)))
with open(data_params.path_to_listops_vocab, 'w') as f:
  json.dump(vocab, f)
