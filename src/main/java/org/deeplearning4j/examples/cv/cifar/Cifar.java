package org.deeplearning4j.examples.cv.cifar;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.canova.image.loader.CifarLoader;
import org.canova.image.transform.FlipImageTransform;
import org.canova.image.transform.ImageTransform;
import org.canova.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.cifar.TestModels.BatchNormModel;
import org.deeplearning4j.examples.cv.cifar.TestModels.LRNModel;
import org.deeplearning4j.examples.cv.cifar.TestModels.Model4;
import org.deeplearning4j.examples.cv.cifar.TestModels.QuickModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
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
public class Cifar {
    protected static final Logger log = LoggerFactory.getLogger(Cifar.class);
    protected static int height = 32;
    protected static int width = 32;
    protected static int channels = 3;
    protected static int numTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;
    protected static int numTestExamples = CifarLoader.NUM_TEST_IMAGES;
    protected static int numLabels = CifarLoader.NUM_LABELS;
    protected static int batchSize = 128;

    protected static int seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 5;

    public static final boolean norm = true; // change to true to run BatchNorm model

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork network;
        int normalizeValue = 255;

        System.out.println("Load data...");

        ImageTransform flipTransform = new FlipImageTransform(rng);
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});

        log.info("Build model....");
        if(norm) {
            epochs = 5;
            network = new BatchNormModel(height, width, channels, numLabels, seed, iterations).init();
        } else {
            epochs = 5;
            network = new QuickModel(height, width, channels, numLabels, seed, iterations).init();
//            epochs = 2;
//            network = new LRNModel(height, width, channels, numLabels, seed, iterations).init();
//            epochs = 120;
//            network = new Model4(height, width, channels, outputNum, seed, iterations).init();
        }
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        for(ImageTransform transform: transforms) {
            MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainExamples, new int[]{height, width, channels}, numLabels, transform, normalizeValue, true));
            network.fit(cifar);
        }

        log.info("Evaluate model....");
        CifarDataSetIterator cifarTest = new CifarDataSetIterator(batchSize, numTestExamples, new int[] {height, width, channels}, normalizeValue, false);
        Evaluation eval = network.evaluate(cifarTest);
        System.out.println(eval.stats(true));

        log.info("****************Example finished********************");

    }


}