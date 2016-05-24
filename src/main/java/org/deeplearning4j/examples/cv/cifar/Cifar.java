package org.deeplearning4j.examples.cv.cifar;

import java.io.IOException;
import java.util.Arrays;

import org.canova.image.loader.CifarLoader;
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

/**
 * CIFAR-10 is an image dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset inculdes 60K
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

    public static final boolean norm = false; // change to true to run BatchNorm model - not currently broken
    static {
        //Force Nd4j initialization, then set data type to double:
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        MultipleEpochsIterator cifar;
        int epochs;
        int height = 32;
        int width = 32;
        int channels = 3;
        int numTrainSamples = CifarLoader.NUM_TRAIN_IMAGES;
        int numTestSamples = CifarLoader.NUM_TEST_IMAGES;
        int batchSize = 100;

        int outputNum = 10;
        int iterations = 5;
        int seed = 123;
        int listenerFreq = 100;

        System.out.println("Load data...");

        //setup the network
        MultiLayerNetwork network;
        if(norm) {
            epochs = 1;
            cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainSamples, "TRAIN"));
            network = new BatchNormModel(height, width, outputNum, channels, seed, iterations).init();
        } else {
//            epochs = 1;
            epochs = 6;
//            epochs = 120;
            cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainSamples, new int[] {height, width, channels}, "TRAIN"));
//            network = new QuickModel(height, width, channels, outputNum, seed, iterations).init();
            network = new LRNModel(height, width, channels, outputNum, seed, iterations).init();
//            network = new Model4(height, width, channels, outputNum, seed, iterations).init();
        }
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        network.fit(cifar);

        CifarDataSetIterator cifarTest = new CifarDataSetIterator(batchSize, numTestSamples, new int[] {height, width, channels}, "TEST");
        Evaluation eval = new Evaluation(cifarTest.getLabels());
        while(cifarTest.hasNext()) {
            DataSet testDS = cifarTest.next(batchSize);
            INDArray output = network.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

    }


}