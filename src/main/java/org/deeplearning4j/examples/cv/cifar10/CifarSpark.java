package org.deeplearning4j.examples.cv.cifar10;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.datavec.spark.functions.data.FilesAsBytesFunction;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.TestModels.CifarCaffeModels;
import org.deeplearning4j.examples.cv.TestModels.CifarModeEnum;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.datavec.DataVecByteDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;

import org.nd4j.linalg.dataset.DataSet;
import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;


import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @Deprecated - Old version before May 2016. Many Spark revisions make this outdated.
 * CIFAR-10 - Spark version
 *
 */

@Deprecated
public class CifarSpark {
    protected static final Logger log = LoggerFactory.getLogger(CifarSpark.class);

    protected static final int height = 32;
    protected static final int width = 32;
    protected static final int channels = 3;
    protected static final int numLabels = CifarLoader.NUM_LABELS;
    protected static int batchSize = 32;
    protected static int iterations = 1;
    protected static int seed = 42;
    protected static Random rng = new Random(seed);
    protected static List<String> labels = new CifarLoader().getLabels();

    public static void main(String[] args) throws Exception {
        int listenerFreq = batchSize;
        int epochs = 1;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[6]");
        sparkConf.setAppName("Cifar");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load train data....");
        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(CifarLoader.fullDir.toString());
        JavaPairRDD<Text, BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> trainData = filesAsBytes.mapToPair(
                new DataVecByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

        JavaRDD<DataSet> train = trainData.map(new Function<Tuple2<Double, DataSet>, DataSet>() {
            @Override
            public DataSet call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._2();
            }
        });

        train.cache();

        log.info("Build model....");


        MultiLayerNetwork network = new CifarCaffeModels(
                height,
                width,
                channels,
                numLabels,
                seed,
                iterations,
                null,
                new int[]{32, 32, 64},
                "relu",
                WeightInit.XAVIER,
                OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
                Updater.ADAM,
                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD,
                1e-3,
                2e-3,
                true,
                4e-3,
                0.9).buildNetwork(CifarModeEnum.OTHER);

        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Setup parameter averaging
        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, network, trainMaster);

        log.info("Train model...");

//        for (int i = 0; i < epochs; i++) {
        sparkNetwork.fit(train);
//        System.out.println("----- Epoch " + i + " complete -----");
//        }

        train.unpersist();

        sparkData = sc.binaryFiles(CifarLoader.fullDir.toString());
        filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> testData = filesAsBytes.mapToPair(
                new DataVecByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

        JavaRDD<DataSet> test = testData.map(new Function<Tuple2<Double, DataSet>, DataSet>() {
            @Override
            public DataSet call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._2();
            }
        });

        test.cache();

        log.info("Eval model...");

        Evaluation evalActual = sparkNetwork.evaluate(test, labels);
        log.info(evalActual.stats());
        test.unpersist();

        log.info("****************Example finished********************");


    }
}