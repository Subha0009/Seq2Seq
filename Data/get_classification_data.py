import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.datasets import imdb

def get_agnews_data(MAXLEN, BATCH_SIZE):
  train_csv = pd.read_csv(PARAMS.PATH_TO_TRAIN_DATA)
  test_csv = pd.read_csv(PARAMS.PATH_TO_TEST_DATA)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=MAXLEN)
  train_data_desc = tokenizer(list(train_csv['Description']), padding='max_length', truncation=True, return_tensors="tf").input_ids 
  test_data_desc = tokenizer(list(test_csv['Description']), padding='max_length', truncation=True, return_tensors="tf").input_ids
  train_data_labels = list(train_csv['Class Index'].map(lambda x: x-1))
  test_data_labels = list(test_csv['Class Index'].map(lambda x: x-1))
  train_data = tf.data.Dataset.from_tensor_slices((train_data_desc, train_data_labels)).shuffle(len(train_csv), reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=False)
  test_data = tf.data.Dataset.from_tensor_slices((test_data_desc, test_data_labels)).shuffle(len(test_csv), reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=False)
  return len(tokenizer), train_data, test_data

def get_imdb_data(MAXLEN, BATCH_SIZE):
  (X_train, y_train), (X_test, y_test) = imdb.load_data(start_char=1,
                                                      oov_char=2,
                                                      index_from=3,
                                                      num_words=20000)

  X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAXLEN, padding='post')
  X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAXLEN, padding='post')
  train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=False)
  test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE, drop_remainder=False)
  return 20002, train_data, test_data
