import numpy as np
import sys
import argparse
import os
import tempfile
import random

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Test signum layer. \
            Error is mean(abs(DifferenceOfOutputs))/mean(abs(ExpectedOutput)).")

    parser.add_argument('-tn', '--tests-number',
                        type=int,
                        default=1,
                        help="Number of tests (default 1)")
    parser.add_argument('-p', '--params',
                        help="Syntax: Input shape: H,W,C")
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

    for i in range(args.tests_number):
        tmpNetProto = tempfile.NamedTemporaryFile()
        tmpNetProto.write(createSignumNet(args.params))
        tmpNetProto.flush()
        net = caffe.Net(tmpNetProto.name, caffe.TEST)
        tmpNetProto.close()
        sys.stdout.write("{}. ".format(i + 1))
        if not args.verbose:
            sys.stdout.write("Input shape: {},{},{}\n".format(net.blobs['input'].data.shape[2],
                                                             net.blobs['input'].data.shape[3],
                                                             net.blobs['input'].data.shape[1]))
        forwardError, diffError = testSignumLayer(net, args)
        forwardErrorSum += forwardError
        diffErrorSum += diffError

    meanForwardError = forwardErrorSum / args.tests_number
    meanDiffError = diffErrorSum / args.tests_number

    print ("\n#############################################################")
    print ("Number of tests: {}\n".format(args.tests_number))
    print ("Mean forward error: {}".format(meanForwardError))
    print ("Mean diff error: {}".format(meanDiffError))
    print ("#############################################################")

def testSignumLayer(net, args):
    if args.verbose:
        stdOut = sys.stdout
    else:
        stdOut = open(os.devnull, 'w')

    stdOut.write("\n#############################################################\n")
    stdOut.write("Forward test\n")
    stdOut.write("Input data:\n")
    randomData = np.random.random_sample(net.blobs['input'].data.shape) * 3 - 1.5
    randomData = np.array(randomData, dtype=np.float32)
    stdOut.write("{}\n".format(randomData))
    net.blobs['input'].data[...] = randomData
    net.forward()
    ones = np.ones(net.blobs['input'].data.shape)
    npOut = np.copysign(ones, net.blobs['input'].data)
    forwardError = computeError(npOut, net.blobs['signum'].data, args)

    stdOut.write("\n#############################################################\n")
    stdOut.write("Diff test\n")
    stdOut.write("Input diff:\n")
    randomDiff = np.random.random_sample(net.blobs['signum'].data.shape) * 3 - 1.5
    randomDiff = np.array(randomDiff, dtype=np.float32)
    stdOut.write("{}\n".format(randomDiff))
    net.blobs['signum'].diff[...] = randomDiff
    net.backward()
    cond = np.logical_and(randomData > -1.0, randomData < 1.0)
    npOut[...] = randomDiff
    npOut[np.invert(cond)] = 0.0
    diffError = computeError(npOut, net.blobs['input'].diff, args)

    stdOut.write("\n#############################################################\n")
    stdOut.write("Forward error: {}\n".format(forwardError))
    stdOut.write("Diff error: {}\n".format(diffError))
    stdOut.write("#############################################################\n")

    if not args.verbose:
        sys.stdout.write("Forward error: {} | ".format(forwardError))
        sys.stdout.write("Diff error: {}\n".format(diffError))

    return forwardError, diffError


def computeError(expectedOutput, signumOutput, args):
    if args.verbose:
        stdOut = sys.stdout
    else:
        stdOut = open(os.devnull, 'w')

    stdOut.write("Output of numpy:\n")
    stdOut.write("{}\n".format(expectedOutput))
    stdOut.write("Output of signum layer:\n")
    stdOut.write("{}\n".format(signumOutput))

    stdOut.write("Difference of outputs:\n")
    difference = np.abs(expectedOutput - signumOutput)
    stdOut.write("{}\n".format(difference))
    stdOut.write("Error:\n")
    error = difference.mean() / np.abs(expectedOutput).mean()
    stdOut.write("{}\n".format(error))

    return error


def createSignumNet(params):
    import caffe
    if params is not None:
        params = params.split()
        inputParams = params[2].split(",")
        height = int(inputParams[0])
        width = int(inputParams[1])
        channels = int(inputParams[2])
    else:
        channels = random.randint(1, 128)
        height = random.randint(1, 128)
        width = random.randint(1, 128)

    net = caffe.NetSpec()
    net.input = caffe.layers.Input(shape=dict(dim=[1, channels, height, width]))
    net.signum = caffe.layers.Signum(net.input)

    return "force_backward: true\n" + str(net.to_proto())


if __name__ == "__main__":
    main()
