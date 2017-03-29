import caffe
import numpy as np
import sys
import argparse
import cv2

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Test signum layer")

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode (default gpu mode)')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if args.cpu:
        print("CPU mode set")
        caffe.set_mode_cpu()
    else:
        print("GPU mode set")
        caffe.set_mode_gpu()

    forwardTestPassed = False
    backwardTestPassed = False

    net = caffe.Net('deploy_signum_layer.prototxt', caffe.TEST)

    print("*********************************************")

    print("Forward test")
    print("Input data:")
    randomInput = np.random.random_sample(net.blobs['signum-input'].data.shape) * 3 - 1
    randomInput[randomInput == 0.0] = -.05326
    randomInput = np.array(randomInput, dtype=np.float32)
    print randomInput
    net.blobs['signum-input'].data[...] = randomInput
    net.forward()
    print("Output of signum layer:")
    print(net.blobs['signum'].data)
    print("Output of numpy signum:")
    numpySignumOutput = np.sign(randomInput)
    print(numpySignumOutput)
    print("Difference of outputs:")
    difference = net.blobs['signum'].data - numpySignumOutput
    print(difference)
    print("Sum of differences:")
    difference = difference.sum()
    print(difference)
    if difference == 0.0:
        print("Forward test PASSED")
        forwardTestPassed = True
    else:
        print("Forward test FAILED")

    print("*********************************************")

    print("Backward test")
    print("Input data:")
    print randomInput
    randomDiff = np.random.random_sample(net.blobs['signum-input'].data.shape) * 3 - 1
    randomDiff[randomDiff == 0.0] = -.05326
    randomDiff = np.array(randomDiff, dtype=np.float32)
    print("Input diff:")
    print randomDiff
    net.blobs['signum'].diff[...] = randomDiff
    net.backward()
    print("Output of signum layer:")
    print(net.blobs['signum-input'].diff)
    print("Output of numpy:")
    randomDiff[randomInput >= 1.0] = 0.0
    randomDiff[randomInput <= -1.0] = 0.0
    print randomDiff
    print("Differences of outputs:")
    difference = net.blobs['signum-input'].diff - randomDiff
    print(difference)
    print("Sum of differences:")
    difference = difference.sum()
    print(difference)
    if difference == 0.0:
        print("Backward test PASSED")
        backwardTestPassed = True
    else:
        print("Backward test FAILED")

    print("*********************************************")

    if forwardTestPassed and backwardTestPassed:
        if args.cpu:
            print("Test of CPU signum layer PASSED")
        else:
            print("Test of GPU signum layer PASSED")
    else:
        if args.cpu:
            print("Test of CPU signum layer FAILED")
        else:
            print("Test of GPU signum layer FAILED")


if __name__ == "__main__":
    main()
