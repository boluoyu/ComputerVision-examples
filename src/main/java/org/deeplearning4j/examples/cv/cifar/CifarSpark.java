package org.deeplearning4j.examples.cv.cifar;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.input.PortableDataStream;
import org.canova.image.loader.CifarLoader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.cifar.TestModels.LRNModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.canova.CanovaByteDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.spark.api.java.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * CIFAR-10 - Spark version
 *
 * Not working due to recent changes in core
 */

public class CifarSpark {
    protected static final Logger log = LoggerFactory.getLogger(CifarSpark.class);

    protected static final int HEIGHT = 32;
    protected static final int WIDTH = 32;
    protected static final int CHANNELS = 3;
    protected static final int numLabels = CifarLoader.NUM_LABELS;
    protected static int batchSize = 32;
    protected static int iterations = 1;
    protected static int seed = 42;
    protected static Random rng = new Random(seed);
    protected static int nWorkers = 6;
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
        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(CifarLoader.TRAINPATH.toString());
        JavaPairRDD<Text, BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> trainData = filesAsBytes.mapToPair(
                new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

        JavaRDD<DataSet> train = trainData.map(new Function<Tuple2<Double, DataSet>, DataSet>() {
            @Override
            public DataSet call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._2();
            }
        });

        train.cache();

        log.info("Build model....");
        MultiLayerNetwork network = new LRNModel(HEIGHT, WIDTH, CHANNELS, numLabels, seed, iterations).init();
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Setup parameter averaging
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, network,tm);

        log.info("Train model...");

//        for (int i = 0; i < epochs; i++) {
        sparkNetwork.fit(train);
//        System.out.println("----- Epoch " + i + " complete -----");
//        }

        train.unpersist();

        sparkData = sc.binaryFiles(CifarLoader.TESTPATH.toString());
        filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> testData = filesAsBytes.mapToPair(
                new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

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