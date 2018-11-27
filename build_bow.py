# AUTHOR : Yuhang Lu
#-*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np
import re



def load_wl():
	f = open('./data/fre_>8_new.dic')
	con = f.read()
	word_list = eval(con)
	wl = {}
	for i,w in enumerate(word_list):
		wl[w] = i
	return wl
def main():
	wl = load_wl()
	now = 0
	for file in open('./data/total_txt_img_cat.list'):
		arr = np.zeros([len(wl),10])
		lab = file.split('\t')[2]
		count = 0
		id_index = np.zeros([len(wl)],dtype='int32')
		content = open('./data/text_seqs/'+file.split()[0]+'.xml').read().split('\n')
		total_lines = len(content)
		for idx,line in enumerate(content):
			seq = line.split('\t')[0].split()
			for word in seq:
				punc = '[,.!\'%*+-/=><]'
				word = re.sub(punc, '', word)
				if word in wl:
					w_id = wl[word]
					# arr[w_id][0] += 1
					if id_index[w_id] < 10:
						arr[w_id][id_index[w_id]] = (idx+1)/total_lines
						id_index[w_id] += 1
					count += 1
		print('now:',now,'count:',count,'shape:',arr.shape)
		s = str(now)
		while len(s)<4:
			s = '0' + s
		np.save('./text_bow_pos/text_'+s+'_'+lab.strip()+'.npy',arr)
		now += 1
		# break
		# 	print labels

if __name__ == '__main__':
	main()
