#-*- coding:utf-8 -*-


import tensorflow as tf
from utils import InputHelper
from model import BiRNN
import numpy as np
import re
# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 25, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 25, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 15, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 100, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_string('pre_trained_vec', None, 'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def load_wl():
	f = open('./data/fre_>10_new.dic')
	con = f.read()
	word_list = eval(con)
	wl = {}
	for i,w in enumerate(word_list):
		wl[w] = i
	return wl
def main():
	data_loader = InputHelper()
	data_loader.create_dictionary(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.data_dir+'/')
	FLAGS.vocab_size = data_loader.vocab_size
	FLAGS.n_classes = data_loader.n_classes
	wl = load_wl()
	# Define specified Model
	model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,	
		vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
		n_classes=FLAGS.n_classes, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		now = 0
		for file in open('./data/total_txt_img_cat.list'):
			word_list = {}
			arr = np.zeros([len(wl),50])
			lab = file.split('\t')[2]
			for line in open('./data/text_seqs/'+file.split()[0]+'.xml'):
				seq = line.split('\t')[0]
				x,w = data_loader.transform_raw(seq, FLAGS.sequence_length)
				_ , out_features = model.inference(sess, data_loader.labels, [x])
				for i,j in enumerate(w):
					punc = '[,.!\'%*+-/=><]'
					j = re.sub(punc, '', j)
					if j in word_list:
						word_list[j] += out_features[i]
					else:
						word_list[j] = out_features[i]
			count = 0
			for w in word_list:
				if w in wl:
					arr[wl[w]] = word_list[w]
					count += 1
			print('now:',now,'count:',count,'shape:',arr.shape)
			s = str(now)
			while len(s)<4:
				s = '0' + s
			np.save('./text_lstm/text_'+s+'_'+lab.strip()+'.npy',arr)
			now += 1
		# 	print labels

if __name__ == '__main__':
	main()