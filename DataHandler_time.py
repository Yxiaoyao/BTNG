import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
import tensorflow as tf


if args.data == 'yelp':
	predir = 'Datasets/Yelp/'
	behs = ['tip', 'neg', 'neutral', 'pos']
elif args.data == 'ml10m':
	predir = 'Datasets/MultiInt-ML10M/'
	behs = ['neg', 'neutral', 'pos']
elif args.data == 'retail':
	if args.target == 'buy':
		predir = 'Datasets/retail/'
	behs = ['pv', 'fav', 'cart', 'buy']
trnfile = predir + 'trn_'
tstfile = predir + 'tst_'


def helpInit(a, b, c):
	ret = [[None] * b for i in range(a)]
	for i in range(a):
		for j in range(b):
			ret[i][j] = [None] * c
	return ret

def timeProcess(trnMats):
	mi = 1e15
	ma = 0
	for i in range(len(trnMats)):
		minn = np.min(trnMats[i].data)
		maxx = np.max(trnMats[i].data)
		mi = min(mi, minn)
		ma = max(ma, maxx)
	maxTime = 0
	for i in range(len(trnMats)):
		newData = ((trnMats[i].data - mi) / (3600*24*args.slot)).astype(np.int32)
		maxTime = max(np.max(newData), maxTime)
		trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
		# csr_matrix 存储稀疏矩阵
	print('MAX TIME', maxTime)
	return trnMats, maxTime + 1

# behs = ['buy']

def ObtainIIMats(trnMats, predir):
	with open(predir+'iiMats', 'rb') as fs:
		iiMats = pickle.load(fs)
	# iiMats = iiMats[3:]# + iiMats[2:]
	return iiMats

def apply_attention(mat, attention_weights):
	"""
	对稀疏矩阵 mat 应用注意力权重。
    attention_weights: 一个与 mat 相同形状的稀疏矩阵或权重向量。
	返回加权后的稀疏矩阵。
	"""
	return mat.multiply(attention_weights)  # 对每个非零元素应用注意力权重

# 为不同的行为类型（beh）设置权重
def get_attention_weights(beh, mat_shape):
    """
    根据行为类型（beh）为数据矩阵生成注意力权重。
    返回一个与 mat_shape 形状相同的稀疏矩阵或稀疏向量。
    """
    if beh == 'pos':
        # 为 'pos' 类型的行为设置较高的权重
        return csr_matrix(np.ones(mat_shape[1]) * 1.2)  # 例如，pos行为的权重为1.5
    elif beh == 'neutral':
        # 为 'neutral' 类型的行为设置较低的权重
        return csr_matrix(np.ones(mat_shape[1]) * 1.0)  # neutral行为的权重为1.2
    else:
        # 其他行为使用默认权重
        return csr_matrix(np.ones(mat_shape[1]) * 0.8)  # 默认权重为1.0


def time_imput_layer(mat):
	# 1. 获取每个用户的交互物品的索引（即非零元素的列索引）
	user_interactions = [mat.indices[mat.indptr[i]:mat.indptr[i + 1]] for i in range(mat.shape[0])]

	# 2. 对每个用户的交互序列进行填充（padding）或截断
	max_sequence_length = 10  # 假设最大序列长度为10
	padded_sequences = []

	for seq in user_interactions:
		# 填充或截断每个用户的交互序列
		if len(seq) > max_sequence_length:
			padded_sequences.append(seq[:max_sequence_length])  # 截断
		else:
			padded_sequences.append(
				np.pad(seq, (max_sequence_length - len(seq), 0), 'constant', constant_values=0))  # 填充

	# 3. 将填充后的交互序列转换为 TensorFlow 的张量
	item_sequence_input = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length])

	# 4. 使用这些交互序列输入到模型中
	# 假设数据准备好了，可以将 padded_sequences 转换为 tf.Tensor
	item_sequence_tensor = tf.constant(padded_sequences, dtype=tf.int32)
	return item_sequence_input, item_sequence_tensor

def LoadData():
	trnMats = list()
	# trnMats2 = list()
	for i in range(len(behs)):
		beh = behs[i]
		path = trnfile + beh
		with open(path, 'rb') as fs:
			mat = pickle.load(fs)
			# 从文件对象中读取序列化数据，并将其反序列化为python对象
		# # 1. 根据行为类型生成注意力权重
		# attention_weights = get_attention_weights(beh, mat.shape)
		# # 2. 应用注意力机制对 mat 加权
		# mat_with_attention = apply_attention(mat, attention_weights)
		# # 3. 将加权后的矩阵追加到 trnMats 列表中
		# trnMats.append(mat_with_attention)
		# trnMats2.append(mat)
		trnMats.append(mat)
		if args.target == 'click':
			trnLabel = (mat if i==0 else 1 * (trnLabel + mat != 0))
		elif args.target == 'buy' and i == len(behs) - 1:
			trnLabel = 1 * (mat != 0)

	trnMats, maxTime = timeProcess(trnMats)
	# test set
	path = tstfile + 'int'
	with open(path, 'rb') as fs:
		tstInt = np.array(pickle.load(fs))
	tstStat = (tstInt!=None)
	tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])
	# 改变数组性质但不改变数据

	iiMats = ObtainIIMats(trnMats, predir)

	return trnMats, iiMats, tstInt, trnLabel, tstUsrs, len(behs), maxTime

def negSamp(temLabel, preSamp, sampSize=1000):
	negset = [None] * sampSize
	cur = 0
	for temval in preSamp:
		if temLabel[temval] == 0:
			negset[cur] = temval
			cur += 1
		if cur == sampSize:
			break
	negset = np.array(negset[:cur])
	return negset

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	# 坐标格式的三维矩阵（i,j,v）
	return csr_matrix(coomat.transpose())
	# transpose反转稀疏矩阵的维度

def transToLsts(mat, mask=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.int32)

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.int32)
	return indices, data, shape

def makeCatIiMats(dic, itmnum):
	retInds = []
	for key in dic:
		temLst = list(dic[key])
		for i in range(len(temLst)):
			if args.data == 'tmall' and args.target == 'click':
				div = 50
			else:
				div = 10
			if args.data == 'ml10m' or args.data == 'tmall' and args.target == 'click':
				scdTemLst = list(np.random.choice(range(len(temLst)), len(temLst) // div, replace=False))
			else:
				scdTemLst = range(len(temLst))
			for j in scdTemLst:
				retInds.append([temLst[i], temLst[j]])
	pckLocs = np.random.permutation(len(retInds))[:100000]
	retInds = np.array(retInds, dtype=np.int32)[pckLocs]
	retData = np.array([1] * retInds.shape[0], np.int32)
	return retInds, retData, [itmnum, itmnum]

def makeIiMats(mat):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = list(map(list, zip(coomat.row, coomat.col)))
	uDict = [set() for i in range(shape[0])]
	for ind in indices:
		usr = ind[0]
		itm = ind[1]
		uDict[usr].add(itm)
	retInds = []
	for usr in range(shape[0]):
		temLst = list(uDict[usr])
		for i in range(len(temLst)):
			if args.data == 'tmall' and args.target == 'click':
				div = 50
			else:
				div = 10
			if args.data == 'ml10m' or args.data == 'tmall' and args.target == 'click':
				scdTemLst = list(np.random.choice(range(len(temLst)), len(temLst) // div, replace=False))
			else:
				scdTemLst = range(len(temLst))
			for j in scdTemLst:
				retInds.append([temLst[i], temLst[j]])
	pckLocs = np.random.permutation(len(retInds))[:100000]
	retInds = np.array(retInds, dtype=np.int32)[pckLocs]
	retData = np.array([1] * retInds.shape[0], np.int32)
	return retInds, retData, [shape[1], shape[1]]

def prepareGlobalData(trnMats, trnLabel, iiMats):
	global adjs
	global adj
	global tpadj
	global iiAdjs
	adjs = trnMats
	iiAdjs = list()
	for i in range(len(iiMats)):
		iiAdjs.append(csr_matrix((iiMats[i][1], (iiMats[i][0][:,0], iiMats[i][0][:,1])), shape=iiMats[i][2]))
	adj = trnLabel.astype(np.float32)
	tpadj = transpose(adj)
	adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
	tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
	for i in range(adj.shape[0]):
		for j in range(adj.indptr[i], adj.indptr[i+1]):
			adj.data[j] /= adjNorm[i]
	for i in range(tpadj.shape[0]):
		for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
			tpadj.data[j] /= tpadjNorm[i]

def sampleLargeGraph(pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN):
	global adjs
	global adj
	global tpadj
	global iiAdjs

	def makeMask(nodes, size):
		mask = np.ones(size)
		# 对应形状size的全1数组
		if not nodes is None:
			mask[nodes] = 0.0
		return mask

	def updateBdgt(adj, nodes):
		if nodes is None:
			return 0
		tembat = 1000
		ret = 0
		for i in range(int(np.ceil(len(nodes) / tembat))):
			st = tembat * i
			ed = min((i+1) * tembat, len(nodes))
			temNodes = nodes[st: ed]
			ret += np.sum(adj[temNodes], axis=0)
			# 数组元素求和，axis=0按行求和 axis=1按列求和
		return ret

	def sample(budget, mask, sampNum):
		score = (mask * np.reshape(np.array(budget), [-1])) ** 2
		norm = np.sum(score)
		if norm == 0:
			return np.random.choice(len(score), 1)
		score = list(score / norm)
		arrScore = np.array(score)
		posNum = np.sum(np.array(score)!=0)
		if posNum < sampNum:
			pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
			# np.squeeze删除指定维度（单维度）
			pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
			pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
			# np.concatenate 数组拼接 axis=0对应列拼接，axis=1对应行拼接
		else:
			pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
		return pckNodes

	usrMask = makeMask(pckUsrs, adj.shape[0])
	itmMask = makeMask(pckItms, adj.shape[1])
	itmBdgt = updateBdgt(adj, pckUsrs)
	if pckItms is None:
		pckItms = sample(itmBdgt, itmMask, len(pckUsrs))
		itmMask = itmMask * makeMask(pckItms, adj.shape[1])
	usrBdgt = updateBdgt(tpadj, pckItms)
	for i in range(sampDepth):
		newUsrs = sample(usrBdgt, usrMask, sampNum)
		usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
		newItms = sample(itmBdgt, itmMask, sampNum)
		itmMask = itmMask * makeMask(newItms, adj.shape[1])
		if i == sampDepth - 1:
			break
		usrBdgt += updateBdgt(tpadj, newItms)
		itmBdgt += updateBdgt(adj, newUsrs)
	usrs = np.reshape(np.argwhere(usrMask==0), [-1])
	itms = np.reshape(np.argwhere(itmMask==0), [-1])
	pckAdjs = []
	pckTpAdjs = []
	pckIiAdjs = []
	for i in range(len(adjs)):
		pckU = adjs[i][usrs]
		tpPckI = transpose(pckU)[itms]
		pckTpAdjs.append(tpPckI)
		pckAdjs.append(transpose(tpPckI))
	for i in range(len(iiAdjs)):
		pckI = iiAdjs[i][itms]
		tpPckI = transpose(pckI)[itms]
		pckIiAdjs.append(tpPckI)
	return pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms
