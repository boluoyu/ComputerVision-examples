package org.deeplearning4j.examples.cv.lfw;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.TestModels.AlexNet;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Labeled Faces in the Wild - Spark version
 */

public class LFWSpark {

    protected static final Logger log = LoggerFactory.getLogger(LFWSpark.class);

    protected static final int height = 100; // original is 250
    protected static final int width = 100;
    protected static final int channels = 3;
    protected static final int numLabels = LFWLoader.NUM_LABELS;
    protected static final int numSamples = 50; //LFWLoader.SUB_NUM_IMAGES - 4;
    protected static int batchSize = 10;
    protected static int iterations = 1;
    protected static int seed = 123;
    protected static boolean useSubset = false;
    protected static double splitTrainTest = 0.8;

    public static void main(String[] args) throws Exception {

        int listenerFreq = batchSize;
        int epochs = 1;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("LFW");
        sparkConf.set("spark.driver.maxResultSize", "4G");
//        conf.set(SparkDl4jMultiLayer.ACCUM_GRADIENT, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");
        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[]{height, width, channels}, numLabels, useSubset, new ParentPathLabelGenerator(), true, splitTrainTest, null, 255, new Random(seed));
        List<String> labels = lfw.getLabels();
        List<DataSet> data = new ArrayList<>();
        while(lfw.hasNext()){
            data.add(lfw.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(data);

        log.info("Build model....");
        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();
        network.init();
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
        sparkNetwork.fit(sparkDataTrain);

        log.info("Eval model...");
        lfw = new LFWDataSetIterator(batchSize, numSamples, new int[]{height, width, channels}, numLabels, useSubset,  new ParentPathLabelGenerator(), false, splitTrainTest, null, 255, new Random(seed));
        data = new ArrayList<>();
        while(lfw.hasNext()){
            data.add(lfw.next());
        }
        JavaRDD<DataSet> sparkDataTest = sc.parallelize(data);
        Evaluation evalActual = sparkNetwork.evaluate(sparkDataTest, labels);
        log.info(evalActual.stats());



        log.info("****************Example finished********************");


    }
}