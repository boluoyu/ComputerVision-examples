package org.deeplearning4j.examples.cv.cifar10;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.TestModels.CifarCaffeModels;
import org.deeplearning4j.examples.cv.TestModels.CifarModeEnum;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @Deprecated - In development in dl4j-benchmarks. Once solid solution found it will be moved to examples
 * CIFAR-10
 *
 * Image dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset inculdes 60K
 * tiny RGB images sized 32 x 32 pixels covering 10 classes. There are 50K training images and 10K test images.
 *
 * Use this example to run cifar-10.
 *
 * Reference: https://www.cs.toronto.edu/~kriz/cifar.html
 * Dataset url: https://s3.amazonaws.com/dl4j-distribution/cifar-small.bin
 * Model: https://gist.github.com/mavenlin/e56253735ef32c3c296d
 *
 */
@Deprecated
public class Cifar {
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);
    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
    protected static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    protected static int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    protected static int numLabels = CifarLoader.NUM_LABELS;
    protected static int batchSize = 128;
    protected static int trainBatchSize;
    protected static int testBatchSize;

    protected static int seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs;
    protected static int nCores = 32;

    protected static int[] nIn;
    protected static int[] nOut;
    protected static String activation;
    protected static WeightInit weightInit;
    protected static OptimizationAlgorithm optimizationAlgorithm;
    protected static Updater updater;
    protected static LossFunctions.LossFunction lossFunctions;
    protected static double learningRate;
    protected static double biasLearningRate;
    protected static boolean regularization;
    protected static double l2;
    protected static double momentum;

    public static CifarModeEnum networkType = CifarModeEnum.CAFFE_QUICK;

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network;
        int normalizeValue = 255;

        System.out.println("Load data...");

        ImageTransform flipTransform = new FlipImageTransform(rng);
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});

        log.info("Build model....");

        switch (networkType) {
            case CAFFE_QUICK:
                epochs = 1;
                trainBatchSize = 100;
                testBatchSize = 100;
                nIn = null;
                nOut = new int[]{32, 32, 64, 64};
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-3;
                biasLearningRate = 2e-3;
                regularization = true;
                l2 = 4e-3;
                momentum = 0.9;
                break;
            case CAFFE_FULL_SIGMOID:
                trainBatchSize = 100;
                testBatchSize = 100;
                epochs = 130;
                nIn = null;
                nOut = new int[]{32, 32, 64, 250};
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-4;
                biasLearningRate = 2e-4;
                regularization = true;
                l2 = 4e-3;
                momentum = 0.9;
                break;
            case CAFFE_BATCH_NORM:
                trainBatchSize = 100;
                testBatchSize = 1000;
                epochs = 120;
                nIn = null;
                nOut = new int[]{32, 32, 64};
                activation = "sigmoid";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.NESTEROVS;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1-3;
                biasLearningRate = Double.NaN;
                regularization = false;
                l2 = 0.0;
                momentum = 0.9;
                break;
            case TORCH_NIN:
                trainBatchSize = 128;
                testBatchSize = 128;
                epochs = 300;
                nIn = null;
                nOut = null;
                activation = "relu";
                weightInit = WeightInit.DISTRIBUTION;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.SGD;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-1;
                biasLearningRate = Double.NaN;
                regularization = true;
                l2 = 5e-4;
                momentum = 0.9;
                break;
            case TORCH_VGG:
                trainBatchSize = 128;
                testBatchSize = 128;
                epochs = 300;
                nIn = null;
                nOut = null;
                activation = "relu";
                weightInit = WeightInit.RELU;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.SGD;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-1;
                biasLearningRate = Double.NaN;
                regularization = true;
                l2 = 5e-4;
                momentum = 0.9;
                break;
            default:
                trainBatchSize = 100;
                testBatchSize = 100;
                epochs = 10;
                nIn = null;
                nOut = new int[]{32, 32, 64};
                activation = "relu";
                weightInit = WeightInit.XAVIER;
                optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
                updater = Updater.ADAM;
                lossFunctions = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
                learningRate = 1e-4;
                biasLearningRate = 2e-4;
                regularization = true;
                l2 = 4e-3;
                momentum = 0.9;
        }
        network = new CifarCaffeModels(
                height,
                width,
                channels,
                numLabels,
                seed,
                iterations,
                nIn,
                nOut,
                activation,
                weightInit,
                optimizationAlgorithm,
                updater,
                lossFunctions,
                learningRate,
                biasLearningRate,
                regularization,
                l2,
                momentum).buildNetwork(networkType);

        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

//        System.out.println("Train model...");
//        for(ImageTransform transform: transforms) {
//            MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, transform, normalizeValue, true));
//            network.fit(cifar);
//        }
//
//        log.info("Evaluate model....");
//        CifarDataSetIterator cifarTest = new CifarDataSetIterator(batchSize, numTestExamples, new int[] {height, width, channels}, normalizeValue, false);
//        Evaluation eval = network.evaluate(cifarTest);
//        System.out.println(eval.stats(true));

        log.info("****************Example finished********************");

    }


}