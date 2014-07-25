from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
import Image
 
# 2D Gibbs Sampler
class Gibbs:
    def __init__(self, init_x, init_y, img):
        self.x        = init_x
        self.y        = init_y
        self.t        = 0
        self.img = img
 
    def update(self):
        cdf = []
				
        if(self.t == 0):
            for i in xrange(self.img.size[0]):
                if i == 0:
                    cdf.append(self.img.getpixel((i, self.y)))
                else:
                    cdf.append(cdf[-1] + self.img.getpixel((i, self.y)))
            self.x = Gibbs.rand_pdf(cdf)
        else:
            for i in xrange(self.img.size[1]):
                if i == 0:
                    cdf.append(self.img.getpixel((self.x, i)))
                else:
                    cdf.append(cdf[-1] + self.img.getpixel((self.x, i)))
            self.y = Gibbs.rand_pdf(cdf)
        self.t = 1 - self.t
 
    @staticmethod
    def rand_pdf(cdf):
        r = random.random() * cdf[-1]
        for i in xrange(len(cdf)):
            if r < cdf[i]:
                return i
        return len(cdf)-1

 
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
 
# Gibbs sampler test
def gibbs_test(init_x, init_y, trial):
    img = Image.open("test.bmp")
	
    gibbs = Gibbs(init_x, init_y, img)
    burn = int(trial / 10)
    for i in range(burn):
        gibbs.update()
 
    nbins = 20
    minx = 0.0
    miny = 0.0
    maxx = 200.0
    maxy = 200.0
    hist = Hist2D(minx, miny, maxx, maxy, nbins)
    for i in range(trial):
        hist.set_value(gibbs.x, gibbs.y)
        gibbs.update()
 
    xs = [0.0] * nbins * nbins
    ys = [0.0] * nbins * nbins
    zs = [0.0] * nbins * nbins
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for y in range(nbins):
        for x in range(nbins):
            i = y * nbins + x
            xs[i] = x * hist.spanx + hist.minx
            ys[i] = y * hist.spany + hist.miny
            zs[i] = hist.get(x, y) * 1.0 / trial
 
    ax.scatter3D(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(minx, maxx)
    ax.set_ylim3d(miny, maxy)
    ax.set_zlim3d(0.0, 0.01)
    plt.suptitle('Gibbs sampler: %d samples' % trial, size='18')
    plt.savefig('gibbs_%d.png' % trial)
    plt.show()
	
    # save the image
    result = Image.new("L", (nbins, nbins))
    for x in xrange(nbins):
        for y in xrange(nbins):
            intensity = hist.get(x, y) * 1.0 / trial * 100 * 255
            if intensity > 255:
                intensity = 255
            result.putpixel((x,y), intensity)
    result.save("result.jpg")
 
if __name__=='__main__':
    gibbs_test(0.0, 0.0, 100)