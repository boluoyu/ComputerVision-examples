package org.deeplearning4j.examples.cv.cifar.TestModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Model: https://gist.github.com/mavenlin/e56253735ef32c3c296d
 * Paper: http://arxiv.org/pdf/1312.4400v3.pdf
 */
public class Model2 {

    private int height;
    private int width;
    private int channels = 3;
    private int outputNum;
    private long seed;
    private int iterations;

    public Model2(int height, int width, int outputNum, int channels, long seed, int iterations) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.outputNum = outputNum;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerNetwork init() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.DISTRIBUTION) // consider standard distribution with std .05
                .dist(new GaussianDistribution(0, 0.05))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.1)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .biasLearningRate(0.1*2)
                .lrPolicySteps(100000)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(192)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn2")
                        .nOut(160)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn3")
                        .nOut(96)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .dropOut(0.5) // TODO double check this works
                        .build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn4")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(192)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn5")
                        .stride(1, 1)
                        .nOut(192)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn6")
                        .stride(1, 1)
                        .nOut(192)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .dropOut(0.5)
                        .build())
                .layer(8, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn7")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(192)
                        .build())
                .layer(9, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn8")
                        .stride(1, 1)
                        .nOut(192)
                        .build())
                .layer(10, new ConvolutionLayer.Builder(1, 1)
                        .name("cnn9")
                        .stride(1, 1)
                        .nOut(10)
                        .learningRate(0.1)
                        .biasLearningRate(0.1)
                        .build())
                .layer(11, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{8, 8})
                        .name("pool3")
                        .stride(1, 1)
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels)
                .build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;

    }
}
