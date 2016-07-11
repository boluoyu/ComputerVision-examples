package org.deeplearning4j.examples.cv.TestModels;

/**
 * ImageNet DataMode
 *
 * Defines which dataset between object recognition (CLS) and location identification (DET).
 * Also defines whether its train, cross validation or test phase
 */
public enum CifarModeEnum {
    BATCH_NORM, FULL_SIGMOID, QUICK, OTHER;
}
