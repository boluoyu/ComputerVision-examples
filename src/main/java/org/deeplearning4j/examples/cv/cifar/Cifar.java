package org.deeplearning4j.examples.cv.cifar;

import java.io.IOException;
import java.util.Arrays;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.cifar.TestModels.BatchNormModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Arrays;

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
    public static void main(String[] args) throws IOException {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        int height = 32;
        int width = 32;
        int channels = 3;
        int numTrainSamples = 100;
        int numTestSamples = 100;
        int batchSize = 30;

        int outputNum = 10;
        int iterations = 5;
        int epochs = 5;
        int seed = 123;
        int listenerFreq = 1;

        System.out.println("Load data...");
        MultipleEpochsIterator cifar = new MultipleEpochsIterator(epochs, new CifarDataSetIterator(batchSize, numTrainSamples, "TRAIN"));

        //setup the network
        MultiLayerNetwork network = new BatchNormModel(height, width, outputNum, channels, seed, iterations).init();
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        System.out.println("Train model...");
        network.fit(cifar);

        CifarDataSetIterator cifarTest = new CifarDataSetIterator(batchSize, numTestSamples, "TEST");
        Evaluation eval = new Evaluation(cifarTest.getLabels());
        while(cifarTest.hasNext()) {
            DataSet testDS = cifarTest.next(batchSize);
            INDArray output = network.output(testDS.getFeatureMatrix());
            eval.eval(testDS.getLabels(), output);
        }
        System.out.println(eval.stats());

    }


}