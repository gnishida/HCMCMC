############################################################################
# HCMCMC preliminary simulation
#
# @author Gen Nishida
#
# Usage: python hcmcmc.py <T> <S> <N> <M>
# T: num steps of MCMC
# S: num of characteristics
# N: num of samples at each step
# M: num of raters

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import Image
import ImageOps
import sys
import timeit
from datetime import datetime

############################################################################
# Setting
EPS = 0.0000001
e = 0.01
a = 1
theta = 1
r = 10

############################################################################
# 2D Histogram
class Hist2D:
    def __init__(self, minx, miny, maxx, maxy, nbins):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.nbins = nbins
        self.spanx = (maxx - minx) / nbins
        self.spany = (maxy - miny) / nbins
        self.bins  = [[0] * nbins for i in range(nbins)]
 
    def set_value(self, x, y):
        bx = int((x - self.minx) / self.spanx)
        by = int((y - self.miny) / self.spany)
        if bx >=0 and by >= 0 and bx < self.nbins and by < self.nbins:
            self.bins[bx][by] += 1
 
    def get(self, x, y):
        return self.bins[x][y]
		
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
def grad_desc(x, w, wh, q, niter):
	cnt = 0
	for i in xrange(niter):
		old_E = E(x, w, wh, q)
		x -= e * dEx(x, w, q)
		normalize(x)
		w -= e * dEw(x, w, wh, q)
		if old_E > E(x, w, wh, q) and old_E - E(x, w, wh, q) < EPS: break
		cnt += 1

############################################################################
# grad_desc_test
#
# wh: true weight of raters
# zp: N samples which are variants of the current state
# S: num of characteristics of each sample
# N: num of samples
# M: num of raters
# niter: num of iterations
#
def grad_desc_test(img, wh, zp, S, N, M, niter):		
	# compute ground truth
	xt = np.zeros((N, S))
	for i in xrange(N):
		for s in xrange(S):
			xt[i][s] = img[s].getpixel((zp[i][0], zp[i][1]))[0]
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
	
	# initialize x, w
	x = np.zeros(xt.shape)
	w = np.array(wh)

	grad_desc(x, w, wh, q, niter)

	return x

############################################################################
# Choose the next state according to the conditional probability
def choose_next(p):
	cdf = []
	for i in xrange(np.size(p)):
		if i == 0:
			cdf.append(p[i])
		else:
			cdf.append(cdf[-1] + p[i])
	
	r = random.random() * cdf[-1]
	for i in xrange(len(cdf)):
		if r < cdf[i]:
			return i
	return len(cdf)-1

############################################################################
# Kolmogrov-Smironov test
def KStest(result, S, T, img, ws):
	total_img = np.zeros(S)
	for s in xrange(S):
		for x in xrange(img[s].size[0]):
			for y in xrange(img[s].size[1]):
				total_img[s] += img[s].getpixel((x, y))[0]
	
	# create F()
	F = np.zeros(result.size[0] * result.size[1])
	F_total = 0.0
	for x in xrange(result.size[0]):
		for y in xrange(result.size[1]):
			expected = 0.0
			for s in xrange(S):
				expected += ws[s] * img[s].getpixel((x, y))[0] / total_img[s]
			F_total += expected
			F[x * result.size[1] + y] = F_total
	
	# normalize F()
	for i in xrange(np.size(F)):
		F[i] /= F_total
		
	# create Fn()
	Fn = np.zeros(result.size[0] * result.size[1])
	Fn_total = 0.0
	for x in xrange(result.size[0]):
		for y in xrange(result.size[1]):
			Fn_total += float(result.getpixel((x, y))) / T
			Fn[x * result.size[1] + y] = Fn_total
	
	# compute D
	D = 0.0
	for i in xrange(result.size[0] * result.size[1]):
		if abs(Fn[i] - F[i]) > D:
			D = abs(Fn[i] - F[i])
	
	return D * math.sqrt(T)
		
############################################################################
# Gibbs sampling
#
# N: num of samples
# M: num of raters
# S: num of characteristics of each sample
# D: num of dimension of states
#    Currently only D=2 is supported.
# T: num of MCMC steps
#
def gibbs_sampling(N, M, S, D, T):
	img = []
	for s in xrange(S):
		img.append(Image.open("truth" + str(s) + ".bmp"))
		ImageOps.grayscale(img[-1])

	# set up the desired weight [1, 0]
	ws = np.zeros(S)
	ws[1] = 1
	
	# initialize the state (center)
	z = np.array([int(img[0].size[0]/2), int(img[0].size[1]/2)])
	
	# synthesize wh
	wh = np.zeros((M, S))
	for k in xrange(M):
		total = 0.0
		for s in xrange(S):
			wh[k][s] = random.random()
			total += wh[k][s] ** 2
		for s in xrange(S):
			wh[k][s] /= math.sqrt(total)
	
	# MCMC
	xlist = []
	ylist = []
	hist = Hist2D(0, 0, img[0].size[0], img[0].size[1], img[0].size[0])
	for t in xrange(T):
		for d in xrange(D):
			# record the current state
			hist.set_value(z[0], z[1])
			#print("sampled: " + str(z[0]) + "," + str(z[1]))
			
			# sample N data
			zp = np.zeros((N, D))
			for i in xrange(N):
				for d2 in xrange(D):
					if d2 == d:
						zp[i][d2] = random.randint(0, img[0].size[0]-1)
					else:
						zp[i][d2] = z[d2]

			# find the optimum by gradient descent
			p = grad_desc_test(img, wh, zp, S, N, M, 100)
			
			# choose the next state according to the cond. prob.
			z[d] = zp[choose_next(np.dot(p, ws))][d]
	
	# save the image
	result = Image.new("L", (img[0].size[0], img[0].size[1]))
	avg = 0.0
	for x in xrange(img[0].size[0]):
		for y in xrange(img[0].size[1]):
			avg += float(hist.get(x, y)) / T
	avg /= (img[0].size[0] * img[0].size[1])
	for x in xrange(img[0].size[0]):
		for y in xrange(img[0].size[1]):
			intensity = int(float(hist.get(x, y)) / T / avg * 128.0)
			if intensity > 255: intensity = 255
			result.putpixel((x, y), intensity)
	result.save("result_" + str(T) + ".jpg")
	
    # X^2 test
	total_img = np.zeros(S)
	for s in xrange(S):
		for x in xrange(img[s].size[0]):
			for y in xrange(img[s].size[1]):
				total_img[s] += img[s].getpixel((x, y))[0]
	X2 = 0.0
	for x in xrange(img[0].size[0]):
		for y in xrange(img[0].size[1]):
			expected = 0.0
			for s in xrange(S):
				expected += ws[s] * img[s].getpixel((x, y))[0] / total_img[s]
			if expected == 0.0: continue
			expected *= T

			X2 += (hist.get(x, y) - expected) ** 2 / expected

	print("trial: " + str(T) + " X2: " + str(X2))
	
	# Kolmogorov-Smirnov test
	print("K-S test: " + str(KStest(result, S, T, img, ws)))

############################################################################
# main
#
if __name__=='__main__':
	argvs = sys.argv
	argc = len(argvs)
	if argc < 4:
		print("Usage: python %s <S> <N> <M>" % argvs[0])
		print("S: num of characteristics")
		print("N: num of samples at each step")
		print("M: num of raters")
		quit()
	
	# At each step of MCMC, 20 samples are uniformly selected.
	# Then, 5 raters judge which is better for each comparison task.
	# One characteristic is used.
	# Two low level parameters are used.
	# Max steps of MCMC is 100.
	for T in xrange(100, 1000, 100):
		start = datetime.now()
		gibbs_sampling(int(argvs[2]), int(argvs[3]), int(argvs[1]), 2, T)
		end = datetime.now()
		print("Elapsed: " + str(end - start))