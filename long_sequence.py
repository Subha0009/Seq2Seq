import data_params
import tensorflow as tf
from TransEvolve import models
from TransEvolve import utils
import sys
BATCH_SIZE = sys.argv[2]
MAXLEN = sys.argv[3]
H_DIM = 256
HEAD = 8
DROP_RATE = 0.1
LEARNING_RATE = float(sys.argv[4])
WARMUP = int(sys.argv[5])
TPU = sys.argv[6]
ZONE = sys.argv[7]
PROJECT = sys.argv[8]
if sys.argv[9]=='imdb':
  NUM_CLASSES = 2
elif sys.argv[9]=='listops':
  NUM_CLASSES = 10
DATA_DIR = sys.argv[10]

if 'random' in sys.argv[1]:
  projection_type = 'random'
elif 'full' in sys.argv[1]:
  projection_type = 'full'
else:
  raise ValueError

if '1' in sys.argv[1]:
  NUM_LAYERS_PER_BLOCK = 6
  NUM_BLOCKS = 1
elif '2' in sys.argv[1]:
  NUM_LAYERS_PER_BLOCK = 3
  NUM_BLOCKS = 2
else:
  raise ValueError

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU, zone=ZONE, project=PROJECT)
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

train_data = make_dataset(DATA_DIR, 
                 batch_size=MAXLEN*BATCH_SIZE, 
                 maxlen=MAXLEN, 
                 split='train',
                 num_replicas=strategy.num_replicas_in_sync,
                 static_batch=True)
test_data = make_dataset(DATA_DIR,
                 batch_size=MAXLEN*BATCH_SIZE, 
                 maxlen=MAXLEN, 
                 split='test',
                 num_replicas=strategy.num_replicas_in_sync,
                 static_batch=True)

train_data = strategy.experimental_distribute_dataset(train_data)
test_data = strategy.experimental_distribute_dataset(test_data)

with strategy.scope():
  model = models.ClassificationModel(input_dim=vocab_size,
                                     hidden_dim=H_DIM,
                                     num_head=HEAD,
                                     projection_type=projection_type,
                                     num_layers_per_block=NUM_LAYERS_PER_BLOCK,
                                     num_blocks=NUM_BLOCKS,
                                     num_classes=NUM_CLASSES,
                                     dropout=DROP_RATE)
  learning_rate = utils.LearningRateSchedule(initial_learning_rate = LEARNING_RATE,
                                       hidden_size = H_DIM,
                                       warmup_steps = WARMUP)
  optimizer = tf.keras.optimizers.Adam(learning_rate)

with strategy.scope():
  if sys.argv[9]=='imdb':
    loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
  elif sys.argv[9]=='listops':
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function()
def train_step(inp, labels):
  with tf.GradientTape() as tape:
    predictions = model(inp, training=True)
    loss = loss_obj(labels, predictions)
    scaled_loss = loss/strategy.num_replicas_in_sync
  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))  
  train_accuracy.update_state(labels, predictions)
  train_loss.update_state(scaled_loss)
  return loss

@tf.function()
def distributed_train_step(inp, labels):
  strategy.run(train_step, args=(inp, labels,))

@tf.function()
def validate_step(inp, labels):
  predictions = model(inp, training=False)
  test_accuracy.update_state(labels, predictions)

@tf.function()
def distributed_test_step(inp, labels):
  strategy.run(validate_step, args=(inp, labels,))

for epoch in range(EPOCHS):
  pbar = tf.keras.utils.Progbar(100000, 
                                width=15, interval=0.005, 
                                stateful_metrics=['train_accuracy', 'train_loss'])
  loss = 0.
  cur_steps = 0
  for inp, labels in train_data:
    distributed_train_step(inp, labels)
    cur_steps += 1
    pbar.add(1, values=[("train_loss", train_loss.result()), 
                        ("train_accuracy", train_accuracy.result())])
  train_accuracy.reset_states()

  for inp, labels in test_data:
    distributed_test_step(inp, labels)
  print('Test accuracy {:.4f}'.format(test_accuracy.result()))
  test_accuracy.reset_states()



