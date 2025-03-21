from . import utils
import math


class GrayCurve:
    def __init__(self, dimension, bits):
        """
            dimension: Number of dimensions
            bits: The number of bits per co-ordinate. Total number of points is
            2**(bits*dimension).
        """
        self.dimension, self.bits = dimension, bits

    @classmethod
    def fromSize(self, dimension, size):
        """
            size: total number of points in the curve.
        """
        x = math.log(size, 2)
        bits = x/dimension
        if not bits == int(bits):
            raise ValueError("Size does not fit a square Gray curve.")
        return GrayCurve(dimension, int(bits))

    def __len__(self):
        return 2**(self.bits*self.dimension)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        return self.point(idx)

    def dimensions(self):
        """
            Size of this curve in each dimension.
        """
        return [2**self.bits]*self.dimension

    def index(self, p):
        idx = 0
        iwidth = self.bits * self.dimension
        for i in range(iwidth):
            bitoff = self.bits-(i/self.dimension)-1
            poff = self.dimension-(i%self.dimension)-1
            b = utils.bitrange(p[poff], self.bits, bitoff, bitoff+1) << i
            idx |= b
        return utils.igraycode(idx)

    def point(self, idx):
        idx = utils.graycode(idx)
        p = [0]*self.dimension
        iwidth = self.bits * self.dimension
        for i in range(iwidth):
            b = utils.bitrange(idx, iwidth, i, i+1) << (iwidth-i-1)//self.dimension
            p[i%self.dimension] |= b
        return p
