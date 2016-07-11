package org.deeplearning4j.examples.cv.TestModels;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *Cifar10 Caffe Model Variations
 *
 * Quick Model based on:
 * http://caffe.berkeleyvision.org/gathered/examples/cifar10.html
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt
 *
 * Full Sigmoid Model based on:
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver_lr1.prototxt
 * https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt

 *
 * BatchNorm Model based on:
 *  https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_sigmoid_solver_bn.prototxt
 *  https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_sigmoid_train_test_bn.prototxt
 */
public class CifarCaffeModels {

    private int height;
    private int width;
    private int channels;
    private int numLabels;
    private long seed;
    private int iterations;
    protected int[] nIn;
    protected int[] nOut;
    protected String activation;
    protected WeightInit weightInit;
    protected OptimizationAlgorithm optimizationAlgorithm;
    protected Updater updater;
    protected LossFunctions.LossFunction lossFunctions;
    protected double learningRate;
    protected double biasLearningRate;
    protected boolean regularization;
    protected double l2;
    protected double momentum;

    MultiLayerConfiguration conf;


    public CifarCaffeModels(int height, int width, int channels, int numLabels, long seed,
                            int iterations, int[] nIn, int[] nOut, String activation,
                            WeightInit weightInit, OptimizationAlgorithm optimizationAlgorithm,
                            Updater updater, LossFunctions.LossFunction lossFunctions,
                            double learningRate, double biasLearningRate,
                            boolean regularization, double l2, double momentum) {

        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
        this.optimizationAlgorithm = optimizationAlgorithm;
        this.updater = updater;
        this.lossFunctions = lossFunctions;
        this.learningRate = learningRate;
        this.biasLearningRate = (biasLearningRate == Double.NaN)? learningRate: biasLearningRate;
        this.regularization = regularization;
        this.l2 = l2;
        this.momentum = momentum;
    }


    public MultiLayerConfiguration initQuick() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(nOut[0])
                        .stride(1, 1)
                        .padding(2, 2)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .nOut(nOut[1])
                        .stride(1, 1)
                        .padding(2, 2)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .nOut(nOut[2])
                        .stride(1, 1)
                        .padding(2, 2)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(6, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(nOut[3])
                        .dropOut(0.5)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation("softmax")
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        return builder.build();
    }

    public MultiLayerConfiguration initFull() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                .activation(activation)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(channels)
                        .nOut(nOut[0])
                        .stride(1, 1)
                        .padding(2, 2)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder(1, 5e-05, 0.75).n(3).build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .nOut(nOut[1])
                        .stride(1, 1)
                        .padding(2, 2)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder(1, 5e-05, 0.75).n(3).build())
                .layer(6, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .nOut(nOut[2])
                        .stride(1, 1)
                        .padding(2, 2)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(nOut[3])
                        .dropOut(0.5)
                        .build())
                .layer(9, new OutputLayer.Builder(lossFunctions)
                        .nOut(numLabels)
                        .activation("softmax")
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);
        conf = builder.build();
        return conf;

    }

    public MultiLayerConfiguration initBN() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(1)
                .lrPolicySteps(5000)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nOut(nOut[0])
                        .activation("identity")
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(2, 2)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new BatchNormalization.Builder().build())
                .layer(3, new ActivationLayer.Builder().build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .nOut(nOut[1])
                        .activation("identity")
                        .stride(1, 1)
                        .padding(2, 2)
                        .biasInit(0)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(5, new BatchNormalization.Builder().build())
                .layer(6, new ActivationLayer.Builder().build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(8, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .nOut(nOut[2])
                        .activation("identity")
                        .stride(1, 1)
                        .padding(2, 2)
                        .biasInit(0)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(9, new BatchNormalization.Builder().build())
                .layer(10, new ActivationLayer.Builder().build())
                .layer(11, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(12, new OutputLayer.Builder(lossFunctions)
                        .nOut(numLabels)
                        .activation("softmax")
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        conf = builder.build();
        return conf;
    }

    public MultiLayerConfiguration initOther() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .weightInit(weightInit).dist(new GaussianDistribution(0, 1e-4))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(1)
                .lrPolicySteps(5000)
                .optimizationAlgo(optimizationAlgorithm)
                .learningRate(learningRate).biasLearningRate(biasLearningRate)
                .updater(updater).momentum(momentum)
                .regularization(regularization).l2(l2)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nOut(nOut[0])
                        .activation("identity")
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(2, 2)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new BatchNormalization.Builder().build())
                .layer(3, new ActivationLayer.Builder().build())
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .nOut(nOut[1])
                        .activation("identity")
                        .stride(1, 1)
                        .padding(2, 2)
                        .biasInit(0)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(5, new BatchNormalization.Builder().build())
                .layer(6, new ActivationLayer.Builder().build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(8, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .nOut(nOut[2])
                        .activation("identity")
                        .stride(1, 1)
                        .padding(2, 2)
                        .biasInit(0)
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .layer(9, new BatchNormalization.Builder().build())
                .layer(10, new ActivationLayer.Builder().build())
                .layer(11, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(12, new OutputLayer.Builder(lossFunctions)
                        .nOut(numLabels)
                        .activation("softmax")
                        .dist(new GaussianDistribution(0, 1e-2))
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(height, width, channels);

        conf = builder.build();
        return conf;
    }
    public MultiLayerNetwork buildNetwork(CifarModeEnum networkType) {
        switch (networkType) {
            case QUICK:
                conf = initQuick();
                break;
            case FULL_SIGMOID:
                conf = initFull();
                break;
            case BATCH_NORM:
                conf = initBN();
                break;
            default:
                conf = initOther();
                break;
        }

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}