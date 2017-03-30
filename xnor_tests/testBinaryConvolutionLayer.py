import caffe
import numpy as np
import sys
import argparse
import cv2

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Test binary convolution layer")

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode (default gpu mode)')
    parser.add_argument('-u', '--gradient-update',
                        action='store_true',
                        help='Apply gradient update on weight gradients (run gradient test)' )
    parser.add_argument('-s', '--gradient-scale',
                        action='store_true',
                        help='Multiply weight gradients with kernel size')
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
    gradientTestPassed = False

    convolutionNet = caffe.Net('deploy_convolution_layer.prototxt', caffe.TEST)
    binaryConvolutionNet = caffe.Net('deploy_binary_convolution_layer.prototxt', caffe.TEST)

    convolutionNet.params['convolution'][0].data[...] = binaryConvolutionNet.params['binary_convolution'][0].data

    alfas = np.abs(convolutionNet.params['convolution'][0].data).mean(axis=(1,2,3))
    for i, alfa in enumerate(alfas):
        convolutionNet.params['convolution'][0].data[i] = np.sign(convolutionNet.params['convolution'][0].data[i]) * alfa

    print("*********************************************")

    print("Forward test")
    print("Input data:")

    randomInput = np.random.random_sample(convolutionNet.blobs['convolution_input'].data.shape) * 2 - 1
    randomInput = np.array(randomInput, dtype=np.float32)
    print randomInput

    convolutionNet.blobs['convolution_input'].data[...] = randomInput
    binaryConvolutionNet.blobs['binary_convolution_input'].data[...] = randomInput

    convolutionNet.forward()
    binaryConvolutionNet.forward()

    print("Output of convolution layer with binary weights:")
    print convolutionNet.blobs['convolution'].data
    print("Output of binary convolution layer:")
    print binaryConvolutionNet.blobs['binary_convolution'].data

    print("Difference of outputs:")
    forwardDifference = convolutionNet.blobs['convolution'].data - binaryConvolutionNet.blobs['binary_convolution'].data
    print forwardDifference
    print("Mean of differences:")
    forwardDifference = np.abs(forwardDifference).mean()
    print forwardDifference

    if np.abs(forwardDifference) < 0.000001:
        print("Forward test PASSED")
        forwardTestPassed = True
    else:
        print("Forward test FAILED")

    print("*********************************************")

    print("Backward test")
    print("Input diff:")
    randomDiff = np.random.random_sample(convolutionNet.blobs['convolution'].diff.shape) * 3 - 1
    randomDiff = np.array(randomDiff, dtype=np.float32)

    convolutionNet.blobs['convolution'].diff[...] = randomDiff
    binaryConvolutionNet.blobs['binary_convolution'].diff[...] = randomDiff

    convolutionNet.backward()
    binaryConvolutionNet.backward()

    print("Output of convolution layer with binary weights:")
    print convolutionNet.blobs['convolution_input'].diff
    print("Output of binary convolution layer:")
    print binaryConvolutionNet.blobs['binary_convolution_input'].diff

    print("Difference of outputs:")
    backwardDifference = convolutionNet.blobs['convolution_input'].diff - binaryConvolutionNet.blobs['binary_convolution_input'].diff
    print backwardDifference
    print("Mean of differences:")
    backwardDifference = np.abs(backwardDifference).mean()
    print backwardDifference

    if np.abs(backwardDifference) < 0.000001:
        print("Backward test PASSED")
        backwardTestPassed = True
    else:
        print("Backward test FAILED")

    print("*********************************************")

    if args.gradient_update:
        print("Gradient test")
        kernelSize = convolutionNet.params['convolution'][0].data.shape[1] * \
                     convolutionNet.params['convolution'][0].data.shape[2] * \
                     convolutionNet.params['convolution'][0].data.shape[3]

        for i in range(len(convolutionNet.params['convolution'][0].diff)):
            convolutionNet.params['convolution'][0].diff[i][np.logical_and(convolutionNet.params['convolution'][0].data[i] < 1.0,
                                       convolutionNet.params['convolution'][0].data[i] > -1.0)] \
                                       *= 1.0/kernelSize + alfas[i]
            convolutionNet.params['convolution'][0].diff[i][np.logical_and(convolutionNet.params['convolution'][0].data[i] >= 1.0,
                                       convolutionNet.params['convolution'][0].data[i] <= -1.0)] \
                                       *= 1.0/kernelSize

        if args.gradient_scale:
            for i in range(len(convolutionNet.params['convolution'][0].diff)):
                convolutionNet.params['convolution'][0].diff[i] *= kernelSize

        print("Output of convolution layer with binary weights:")
        print convolutionNet.params['convolution'][0].diff
        print("Output of binary convolution layer:")
        print binaryConvolutionNet.params['binary_convolution'][0].diff

        print("Difference of outputs:")
        gradientDifference = convolutionNet.params['convolution'][0].diff - binaryConvolutionNet.params['binary_convolution'][0].diff
        print gradientDifference
        print("Mean of differences:")
        gradientDifference = np.abs(gradientDifference).mean()
        print gradientDifference

        if np.abs(gradientDifference) < 0.000001:
            print("Gradient test PASSED")
            gradientTestPassed = True
        else:
            print("Gradient test FAILED")

        print("*********************************************")
    else:
        gradientTestPassed = True

    print("Forward test mean of differences: {}".format(forwardDifference))
    print("Backward test mean of differences: {}".format(backwardDifference))
    print("Gradient test mean of differences: {}".format(gradientDifference))

    if forwardTestPassed and backwardTestPassed and gradientTestPassed:
        if args.cpu:
            print("Test of CPU binary convolution layer PASSED")
        else:
            print("Test of GPU binary convolution layer PASSED")
    else:
        if args.cpu:
            print("Test of CPU binary convolution layer FAILED")
        else:
            print("Test of GPU binary convolution layer FAILED")


if __name__ == "__main__":
    main()
