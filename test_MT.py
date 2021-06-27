import data_params
import tensorflow as tf
from TransEvolve import models
from TransEvolve import utils
from Data import tokenizer
from Data import test_data_translation
import sys
H_DIM = 512
HEAD = 8
DROP_RATE = 0.1
checkpoint_path = sys.argv[3]
ckpt_index = sys.argv[4]
if 'en-de' in sys.argv[2]:
  PARAMS = data_params.en_de_params
elif 'en-fr' in sys.argv[2]:
  PARAMS = data_params.en_fr_params

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

vocab_size = len(open(PARAMS["path_to_mt_vocab"]).readlines())
model = models.EncoderDecoderModel(input_dim=vocab_size,
                                     hidden_dim=H_DIM,
                                     num_head=HEAD,
                                     projection_type=projection_type,
                                     num_layers_per_block=NUM_LAYERS_PER_BLOCK,
                                     num_blocks=NUM_BLOCKS,
                                     dropout=DROP_RATE)

optimizer = tf.keras.optimizers.Adam(0.1, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
ckpt = tf.train.Checkpoint(model=model,
                             optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
ckpt.restore(checkpoint_path+'/'+ckpt_index)
print('Checkpoint restored; Model was trained for {} steps.'.format(ckpt.optimizer.iterations.numpy()))
subWordTokenizer = tokenizer.Subtokenizer(PARAMS["path_to_mt_vocab"])
test_dataset = test_data_translation.make_dataset(PARAMS["path_to_src"], PARAMS["path_to_tar"], subWordTokenizer, batch_size=25000)
model.batch_evaluate(test_dataset, vocab_size, subWordTokenizer, alpha=0.6, beam_size=4)
