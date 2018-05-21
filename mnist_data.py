
from __future__ import print_function

import tensorflow as tf
import numpy


# from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data")#, one_hot=True)

import gzip
# f = gzip.open('t10k-images-idx3-ubyte.gz', 'rb')
# file_content = f.read()
#
# for x in range(100):
#     print(file_content[x])

# with gzip.open('train-images-idx3-ubyte.gz','r') as fin:
    # for line in fin:
    #     print('got line', line)
    #     if line == 2:
    #         break
    # print(fin)

with open('t10k-images-idx3-ubyte.gz', 'rb') as f:

    def _read32(bytestream):
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)

# print(data[2][0])
# print("-----")
# print(data[2][1])
# print("-----")
# print(data[2][2])

print(len(data[0]))
# data[0] is an array with 28 elements
# each element is an array of 28 numbers (each element is thus a row)

for y in data[0]:
    for x in y:
        if x > 0:
            print("X", end=' ')
        else:
            print(".", end=' ')
    print(" ")

# arr = [[1, 2], [3, 4]]
#
# for y in arr:
#     for x in y:
#         print(x, end=' ')
#     print(" ")

# with open('t10k-labels-idx1-ubyte.gz', 'rb') as f2:

    # def _read32(bytestream):
    #     dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    #     return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
    #
    # with gzip.GzipFile(fileobj=f2) as bytestream:
    #     magic = _read32(bytestream)
    #     num_images = _read32(bytestream)
    #     rows = _read32(bytestream)
    #     cols = _read32(bytestream)
    #     buf = bytestream.read(rows * cols * num_images)
    #     data = numpy.frombuffer(buf, dtype=numpy.uint8)
    #     data = data.reshape(num_images, rows, cols, 1)

    # print(f2)

# with gzip.open('train-labels-idx1-ubyte.gz','r') as fin:
#     for line in fin:
#         print('-------------------------------------------', line)
