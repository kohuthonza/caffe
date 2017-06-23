import os
import numpy as np
import sys
import argparse
import cv2
import tempfile
import random
from google.protobuf.text_format import Merge


def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Test binary convolution layer. \
                             Error is absDifferenceOfOutputs/expectedOutput.")

    parser.add_argument('-n', '--tests-number',
                        type=int,
                        default=1,
                        help="Number of tests (default 1)")
    parser.add_argument('-u', '--update-weight-diff',
                        action='store_true',
                        help="Apply gradient update on weight gradients.")
    parser.add_argument('-s', '--scale-weight-diff',
                        action='store_true',
                        help="Multiply weight gradients with kernel size.")
    parser.add_argument('-p', '--params',
                        help="Syntax: Input shape: H,W,C  Conv params: kernelNumber,kernelSize,stride,pad")
    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help="Set cpu mode (default gpu mode).")
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help="Print inputs, outputs and differences per single value.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not args.verbose:
        os.environ['GLOG_minloglevel'] = '2'

    import caffe
    from caffe.proto import caffe_pb2

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if args.verbose:
        if args.cpu:
            print("CPU mode set.")
        else:
            print("GPU mode set.")

    forwardErrorSum = 0.0
    diffErrorSum = 0.0
    weightDiffErrorSum = 0.0

    for i in range(args.tests_number):
        tmpNetProto = tempfile.NamedTemporaryFile()
        tmpNetProto.write(createConvolutionsNet(args.params, args.update_weight_diff, args.scale_weight_diff))
        tmpNetProto.flush()
        net = caffe.Net(tmpNetProto.name, caffe.TEST)
        deploy = caffe_pb2.NetParameter()
        Merge((open(tmpNetProto.name,'r').read()), deploy)
        tmpNetProto.close()
        sys.stdout.write("{}. ".format(i + 1))
        forwardError, diffError, weightDiffError = testBinaryConvolutionLayer(net, deploy, args)
        forwardErrorSum += forwardError
        diffErrorSum += diffError
        weightDiffErrorSum += weightDiffError

    meanForwardError = forwardErrorSum / args.tests_number
    meanDiffError = diffErrorSum / args.tests_number
    meanWeightDiffError = weightDiffErrorSum / args.tests_number

    print ("\n#############################################################")
    print ("Number of tests: {}\n".format(args.tests_number))
    print ("Mean forward error: {}".format(meanForwardError))
    print ("Mean diff error: {}".format(meanDiffError))
    print ("Mean weight diff error: {}".format(meanWeightDiffError))
    print ("#############################################################")


def testBinaryConvolutionLayer(net, deploy, args):
    if args.verbose:
        stdOut = sys.stdout
    else:
        stdOut = open(os.devnull, 'w')

    randomWeights = np.random.random_sample(net.params['convolution'][0].data.shape) * 3.0 - 1.5
    randomBiases = np.random.random_sample(net.params['convolution'][1].data.shape) * 0.25 - 0.25
    ones = np.ones(randomWeights[0].shape)
    alfas = np.abs(randomWeights).mean(axis=(1,2,3))
    for i, alfa in enumerate(alfas):
        net.params['convolution'][0].data[i] = np.copysign(ones, randomWeights[i]) * alfa
    net.params['binary_convolution'][0].data[...] = randomWeights
    net.params['convolution'][1].data[...] = randomBiases
    net.params['binary_convolution'][1].data[...] = randomBiases

    stdOut.write("\n#############################################################\n")
    stdOut.write("Forward test\n")
    stdOut.write("Input data:\n")
    randomData = np.random.random_sample(net.blobs['convolution_input'].data.shape) - 0.5
    randomData = np.array(randomData, dtype=np.float32)
    stdOut.write("{}\n".format(randomData))
    net.blobs['convolution_input'].data[...] = randomData
    net.blobs['binary_convolution_input'].data[...] = randomData
    net.forward()
    forwardError = computeError(net.blobs['convolution'].data,
                                net.blobs['binary_convolution'].data,
                                args)

    stdOut.write("\n#############################################################\n")
    stdOut.write("Diff test\n")
    stdOut.write("Input diff:\n")
    randomDiff = np.random.random_sample(net.blobs['convolution'].diff.shape) - 0.5
    randomDiff = np.array(randomDiff, dtype=np.float32)
    stdOut.write("{}\n".format(randomDiff))
    net.blobs['convolution'].diff[...] = randomDiff
    net.blobs['binary_convolution'].diff[...] = randomDiff
    net.backward()
    diffError = computeError(net.blobs['convolution_input'].diff,
                             net.blobs['binary_convolution_input'].diff,
                             args)

    stdOut.write("\n#############################################################\n")
    stdOut.write("Weight diff test\n")
    kernelSize = net.params['convolution'][0].data.shape[1] * \
                 net.params['convolution'][0].data.shape[2] * \
                 net.params['convolution'][0].data.shape[3]
    stdOut.write("Weights:\n")
    stdOut.write("{}\n".format(net.params['convolution']))
    if args.update_weight_diff:
        for i in range(len(net.params['convolution'][0].diff)):
            cond = np.logical_and(net.params['binary_convolution'][0].data[i] < 1.0,
                                  net.params['binary_convolution'][0].data[i] > -1.0)
            net.params['convolution'][0].diff[i][cond] *= 1.0/kernelSize + alfas[i]
            net.params['convolution'][0].diff[i][np.invert(cond)] *= 1.0/kernelSize
    if args.update_weight_diff and args.scale_weight_diff:
        for i in range(len(net.params['convolution'][0].diff)):
            net.params['convolution'][0].diff[i] *= kernelSize
    weightDiffError = computeError(net.params['convolution'][0].diff,
                                   net.params['binary_convolution'][0].diff,
                                   args)

    stdOut.write("\n#############################################################\n")
    stdOut.write("Forward error: {}\n".format(forwardError))
    stdOut.write("Diff error: {}\n".format(diffError))
    stdOut.write("Weight diff error: {}\n".format(weightDiffError))
    stdOut.write("#############################################################\n")

    if not args.verbose:
        sys.stdout.write("Input shape: {},{},{} ".format(net.blobs['convolution_input'].data.shape[2],
                                                         net.blobs['convolution_input'].data.shape[3],
                                                         net.blobs['convolution_input'].data.shape[1]))
        convParams = deploy.layer[2].convolution_param
        sys.stdout.write("Conv params: {},{},{},{} ".format(convParams.num_output,
                                                            convParams.kernel_size[0],
                                                            convParams.stride[0],
                                                            convParams.pad[0]))
        sys.stdout.write("Output shape: {},{},{}\n".format(net.blobs['convolution'].data.shape[2],
                                                           net.blobs['convolution'].data.shape[3],
                                                           net.blobs['convolution'].data.shape[1]))
        sys.stdout.write("Forward error: {} | ".format(forwardError))
        sys.stdout.write("Diff error: {} | ".format(diffError))
        sys.stdout.write("Weight diff error: {}\n".format(weightDiffError))

    return forwardError, diffError, weightDiffError


def computeError(convolutionOutput, binaryConvolutionOutput, args):
    if args.verbose:
        stdOut = sys.stdout
    else:
        stdOut = open(os.devnull, 'w')

    stdOut.write("Output of convolution layer with binary weights:\n")
    stdOut.write("{}\n".format(convolutionOutput))
    stdOut.write("Output of binary convolution layer:\n")
    stdOut.write("{}\n".format(binaryConvolutionOutput))

    stdOut.write("Difference of outputs:\n")
    difference = np.abs(convolutionOutput - binaryConvolutionOutput)
    stdOut.write("{}\n".format(difference))
    stdOut.write("Error:\n")
    error = difference.mean() / np.abs(convolutionOutput).mean()
    stdOut.write("{}\n".format(error))

    return error


def createConvolutionsNet(params, update, scale):
    import caffe
    if params is not None:
        params = params.split()
        inputParams = params[2].split(',')
        height = int(inputParams[0])
        width = int(inputParams[1])
        channels = int(inputParams[2])
        convParams = params[5].split(',')
        kernelNumber = int(convParams[0])
        kernelSize = int(convParams[1])
        stride = int(convParams[2])
        pad = int(convParams[3])
    else:
        kernelNumber = random.randint(1, 256)
        kernelSize = random.randint(1, 5)
        stride = random.randint(1, 5)
        pad = random.randint(0, kernelSize - 1)

        channels = random.randint(1, 128)
        height = random.randint(1, 256)
        width = random.randint(1, 256)
        # Adjust input to exactly fit conv params
        height = adjustDimension(height, kernelSize, stride, pad)
        width = adjustDimension(width, kernelSize, stride, pad)

    net = caffe.NetSpec()
    net.convolution_input = caffe.layers.Input(shape=dict(dim=[1, channels, height, width]))
    net.binary_convolution_input = caffe.layers.Input(shape=dict(dim=[1, channels, height, width]))
    net.convolution = caffe.layers.Convolution(net.convolution_input,
                                               num_output=kernelNumber,
                                               kernel_size=kernelSize,
                                               stride=stride,
                                               pad=pad)
    net.binary_convolution = caffe.layers.BinaryConvolution(net.binary_convolution_input,
                                                            convolution_param=dict(
                                                            num_output=kernelNumber,
                                                            kernel_size=kernelSize,
                                                            stride=stride,
                                                            pad=pad),
                                                            binary_convolution_param=dict(
                                                            update_weight_diff=update,
                                                            scale_weight_diff=scale
                                                            ))

    return "force_backward: true\n" + str(net.to_proto())


def adjustDimension(dimension, kernelSize, stride, pad):
    spaceToMoveKernelDimensionWise = dimension + 2 * pad - kernelSize

    if spaceToMoveKernelDimensionWise < 0:
        dimension -= spaceToMoveKernelDimensionWise
    else:
        dimensionOffset = spaceToMoveKernelDimensionWise % stride
        if (dimensionOffset) != 0:
            dimension += stride - dimensionOffset

    return dimension


if __name__ == "__main__":
    main()
