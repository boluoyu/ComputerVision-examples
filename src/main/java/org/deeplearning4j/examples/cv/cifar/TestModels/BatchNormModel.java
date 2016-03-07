package org.deeplearning4j.examples.cv.cifar.TestModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *  Model: https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt
 */
public class BatchNormModel {
    private int height;
    private int width;
    private int channels = 3;
    private int outputNum;
    private long seed;
    private int iterations;

    public BatchNormModel(int height, int width, int outputNum, int channels, long seed, int iterations) {
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
                .weightInit(WeightInit.DISTRIBUTION) // consider standard distribution with std .05
                .dist(new GaussianDistribution(0, 1e-4))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(1)
                .biasLearningRate(0.1*2)
                .lrPolicySteps(5000)
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
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new BatchNormalization.Builder().build())
                .layer(3, new ActivationLayer.Builder().activation("sigmoid").build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .activation("identity")
                        .biasInit(0)
                        .build())
                .layer(5, new BatchNormalization.Builder().build())
                .layer(6, new ActivationLayer.Builder().activation("sigmoid").build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(8, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(64)
                        .activation("identity")
                        .biasInit(0)
                        .build())
                .layer(9, new BatchNormalization.Builder().build())
                .layer(10, new ActivationLayer.Builder().activation("sigmoid").build())
                .layer(11, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}