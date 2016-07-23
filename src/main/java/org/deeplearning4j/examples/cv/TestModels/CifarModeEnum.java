package org.deeplearning4j.examples.cv.TestModels;

/**
 * ImageNet DataMode
 *
 * Defines which dataset between object recognition (CLS) and location identification (DET).
 * Also defines whether its train, cross validation or test phase
 */
public enum CifarModeEnum {
    CAFFE_BATCH_NORM, CAFFE_FULL_SIGMOID, CAFFE_QUICK, TORCH_NIN, TORCH_VGG, OTHER;
}
