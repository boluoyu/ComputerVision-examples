package org.deeplearning4j.examples.cv.cifar.TestModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Cifar10 LRN
 *
 * Model based on:
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver_lr1.prototxt
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt
 */
public class LRNModel {
    private int height;
    private int width;
    private int channels = 3;
    private int outputNum;
    private long seed;
    private int iterations;

    public LRNModel(int height, int width, int channels, int outputNum, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }
    public MultiLayerNetwork init() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.DISTRIBUTION) // consider standard distribution with std .05
                .dist(new GaussianDistribution(0, 1e-4))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.0001)
                .biasLearningRate(0.0002)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .regularization(true)
                .l2(0.004)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder(1, 5e-05, 0.75).n(3).build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder(1, 5e-05, 0.75).n(3).build())
                .layer(6, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(64)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(250)
                        .dropOut(0.5)
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}