############################################################################
# Gradient descent test for HCMCMC
#
# @author Gen Nishida

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import Image

############################################################################
# Setting
EPS = 0.0000001
e = 0.01
a = 1
theta = 1
M = 10
r = 10

############################################################################
# normalize x
def normalize(x):
	for i in xrange(x.shape[1]):
		avg = np.average(x[:,i])
		x[:,i] -= avg
		m = np.max(np.absolute(x[:,i]))
		if m > 0:
			x[:,i] /= m

############################################################################
# E = ... eq(4)
def E(x, w, wh, q):
	result = np.ndarray.sum((w - wh) * (w - wh)) * 0.5 / a / a
	result += np.ndarray.sum(x * x) * 0.5 / theta / theta
	for i in xrange(x.shape[0]):
		for j in xrange(i+1, x.shape[0]):
			for k in xrange(w.shape[0]):
				result -= (1.0 - q[i][j][k]) * np.dot(w[k], (x[j] - x[i]))
	for i in xrange(x.shape[0]):
		for j in xrange(i+1, x.shape[0]):
			for k in xrange(w.shape[0]):
				result += math.log(1 + math.exp(np.dot(w[k], x[j] - x[i])))
	return result
	
############################################################################
# dExi = .. eq(6)
def dExi(i, x, w, q):
	result = np.array(x[i]) / theta / theta
	for j in xrange(x.shape[0]):
		if j == i: continue
		for k in xrange(w.shape[0]):
			result -= w[k] * q[i][j][k]
	for j in xrange(i+1, x.shape[0]):
		for k in xrange(w.shape[0]):
			result += w[k]
	for j in xrange(i+1, x.shape[0]):
		for k in xrange(w.shape[0]):
			result -= w[k] * math.exp(np.dot(w[k], x[j] - x[i])) / (1.0 + math.exp(np.dot(w[k], x[j] - x[i])))
	for j in xrange(i):
		for k in xrange(w.shape[0]):
			result += w[k] * math.exp(np.dot(w[k], x[i] - x[j])) / (1.0 + math.exp(np.dot(w[k], x[i] - x[j])))
	return result

############################################################################
# dEx = .. eq(6)
def dEx(x, w, q):
	result = np.zeros(x.shape)
	for i in xrange(result.shape[0]):
		result[i] = dExi(i, x, w, q)
	return result

############################################################################
# dEwk = .. eq(7)
def dEwk(k, x, w, wh, q):
	result = np.array(w[k] - wh[k]) / a / a
	for i in xrange(x.shape[0]):
		for j in xrange(i+1, x.shape[0]):
			result -= (1 - q[i][j][k]) * (x[j] - x[i])
	for i in xrange(x.shape[0]):
		for j in xrange(i+1, x.shape[0]):
			result += (x[j] - x[i]) * math.exp(np.dot(w[k], x[j] - x[i])) / (1.0 + math.exp(np.dot(w[k], x[j] - x[i])))
	return result

############################################################################
# dEw = .. eq(7)
def dEw(x, w, wh, q):
	result = np.zeros(w.shape)
	for k in xrange(result.shape[0]):
		result[k] = dEwk(k, x, w, wh, q)
	return result

############################################################################
# gradient descent
#
def grad_desc(x, w, wh, q, T):
	cnt = 0
	for i in xrange(T):
		old_E = E(x, w, wh, q)
		x -= e * dEx(x, w, q)
		normalize(x)
		w -= e * dEw(x, w, wh, q)
		if old_E > E(x, w, wh, q) and old_E - E(x, w, wh, q) < EPS: break
		cnt += 1

	print("num steps: " + str(cnt))

############################################################################
# grad_desc_test
#
# D: dimension of the state
#    Since we use only one image as a probability distribution,
#    only D=1 is supported.
# N: num of state
# M: num of raters
# T: num of steps
#
def grad_desc_test(D, N, M, T):
	img = []
	for d in xrange(D):
		img.append(Image.open("truth" + str(d) + ".bmp"))
	
	# synthesize wh
	wh = np.zeros((M, D))
	for k in xrange(M):
		total = 0.0
		for d in xrange(D):
			wh[k][d] = random.random()
			total += wh[k][d] ** 2
		for d in xrange(D):
			wh[k][d] /= math.sqrt(total)

	# sample N data
	zp = np.zeros((N, 2))
	for i in xrange(N):
		zp[i][0] = 100
		zp[i][1] = random.randint(0, 199)

	# compute ground truth
	xt = np.zeros((N, D))
	for i in xrange(N):
		for d in xrange(D):
			xt[i][d] = img[d].getpixel((zp[i][0], zp[i][1]))
	normalize(xt)
						
	# TODO
	# We should add Gaussian to xt before synthesize q

	# synthesize q
	q = np.zeros((N, N, M))
	for i in xrange(N):
		for j in xrange(i+1, N):
			for k in xrange(M):
				if random.random() <= 1.0 / (1.0 + math.exp(r * np.dot(wh[k], xt[j] - xt[i]))):
				#if np.dot(wh[k], xt[i] - xt[j]) > 0:
					q[i][j][k] = 1
				q[j][i][k] = 1 - q[i][j][k]
				
	print("----------------------------------------")
	print("ground truth of x:")
	print(xt)
	print("ground truth of w:")
	print(wh)
	#print("votes:")
	#print(q)
	
	# initialize x, w
	x = np.zeros(xt.shape)
	w = np.array(wh)

	grad_desc(x, w, wh, q, T)

	# display the results
	print("----------------------------------------")
	print("estimate of x:")
	print(x)
	print("estimate of w:")
	print(w)
	
	# evaluate x against q
	correct = 0
	incorrect = 0
	for i in xrange(N):
		for j in xrange(i+1, N):
			for k in xrange(M):
				if np.dot(w[k], x[i] - x[j]) > 0.0:
					if q[i][j][k] == 1:
						correct += 1
					else:
						incorrect += 1
						#print("------------------")
						#print("x[" + str(i) + "]: " + str(x[i]))
						#print("x[" + str(j) + "]: " + str(x[j]))
						#print("true x[" + str(i) + "]: " + str(xt[i]))
						#print("true x[" + str(j) + "]: " + str(xt[j]))
				else:
					if q[i][j][k] == 1:
						incorrect += 1
					else:
						correct += 1
	print("comparison test:")
	print("correct: " + str(correct))
	print("incorrect: " + str(incorrect))
	
	
	# test
	correct = 0
	incorrect = 0
	for i in xrange(N):
		for j in xrange(i+1, N):
			for d in xrange(D):
				if x[i][d] > x[j][d]:
					if xt[i][d] > xt[j][d]:
						correct += 1
					else:
						incorrect += 1
				else:
					if xt[i][d] > xt[j][d]:
						incorrect += 1
					else:
						correct += 1
	print("per component test:")
	print("correct: " + str(correct))
	print("incorrect: " + str(incorrect))
	
	return x
	
	
if __name__=='__main__':
	# pick 10 samples, ask 5 raters and get the conditional distribution
	grad_desc_test(2, 10, 5, 100)
