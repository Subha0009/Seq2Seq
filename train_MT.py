import data_params
import tensorflow as tf
from TransEvolve import models
from TransEvolve import utils
from Data import translation_pipeline
import sys
BATCH_SIZE = int(sys.argv[3])
MAXLEN = int(sys.argv[4])
H_DIM = 512
HEAD = 8
DROP_RATE = 0.1
LEARNING_RATE = float(sys.argv[5])
WARMUP = int(sys.argv[6])
TPU = sys.argv[7]
ZONE = sys.argv[8]
PROJECT = sys.argv[9]
checkpoint_path = sys.argv[11]
DATA_DIR = sys.argv[12]
if 'en-de' in sys.argv[10]:
  PARAMS = data_params.en_de_params
elif 'en-fr' in sys.argv[10]:
  PARAMS = data_params.en_fr_params
elif 'de-en' in sys.argv[10]:
    PARAMS = data_params.de_en_params

if 'random' in sys.argv[2]:
  projection_type = 'random'
elif 'full' in sys.argv[2]:
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

if 'learned' in sys.argv[13]:
  learned_pos_encoding=True
else:
  learned_pos_encoding=False

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU, zone=ZONE, project=PROJECT)
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

train_data = strategy.experimental_distribute_dataset(
                        translation_pipeline.train_input_fn(data_dir=DATA_DIR,
                                                    batch_size=BATCH_SIZE*MAXLEN,
                                                    max_length=MAXLEN,
                                                    max_io_parallelism=40,
                                                    repeat_dataset=1,
                                                    static_batch=True,
                                                    num_gpus=strategy.num_replicas_in_sync,
                                                    ctx=None))
vocab_size = len(open(PARAMS["path_to_mt_vocab"]).readlines())
with strategy.scope():
  model = models.EncoderDecoderModel(input_dim=vocab_size,
                                     hidden_dim=H_DIM,
                                     num_head=HEAD,
                                     projection_type=projection_type,
                                     num_layers_per_block=NUM_LAYERS_PER_BLOCK,
                                     num_blocks=NUM_BLOCKS,
                                     dropout=DROP_RATE,
                                     learned_pos_encoding=learned_pos_encoding)

  learning_rate = utils.LearningRateSchedule(initial_learning_rate = LEARNING_RATE,
                                       hidden_size = H_DIM,
                                       warmup_steps = WARMUP)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
  ckpt = tf.train.Checkpoint(model=model,
                             optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
  else:
    print('Training from scratch!')

def train_step(inp, tar):
  tar_in = tar[:, :-1]
  tar_out = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions = model(inp, tar_in, training=True)
    loss = utils.padded_cross_entropy_loss(predictions, tar_out)
  tvars = list({id(v): v for v in model.trainable_variables}.values())
  grads = tape.gradient(loss, tvars)
  optimizer.apply_gradients(zip(grads, tvars))
  
  train_loss.update_state(loss)
  
@tf.function()
def distributed_train_step(src_data, tar_data):
  strategy.run(train_step, args=(src_data, tar_data,))

EPOCHS = 30
target = 300000
pbar = tf.keras.utils.Progbar(target, width=15, interval=0.005, stateful_metrics=['train_loss'])
steps = optimizer.iterations.numpy()
pbar.add(steps, values=[("train_loss", train_loss.result())])
for epoch in range(EPOCHS):
  for inp, tar in train_data:
    distributed_train_step(inp, tar)
    pbar.add(1, values=[("train_loss", train_loss.result())])  
    steps += 1
    if (steps>150000 and steps%2000==0) or steps%10000==0:
      ckpt_save_path = ckpt_manager.save()
  train_loss.reset_states()
