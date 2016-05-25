package org.deeplearning4j.examples.cv.msra_cfw;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.AlexNet;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * WORK IN PROGRESS Face Classification Spark Version
 *
 * Checkout the basic MSRA_CFW for more information.
 *
 * Not working due to recent changes in core
 *
 */

public class MSRA_CFWSpark {
    protected static final Logger log = LoggerFactory.getLogger(MSRA_CFWSpark.class);

    public final static int NUM_IMAGES = 2215; // some are 50 and others 700
    public final static int NUM_LABELS = 10;
    public final static int HEIGHT = 40;
    public final static int WIDTH = 40; // size varies
    public final static int CHANNELS = 3;

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numExamples",usage="Number of examples",aliases="-nE")
    protected int numExamples = 100;
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 1;
    @Option(name="--epochs",usage="Number of epochs",aliases="-ep")
    protected int epochs = 5;
    @Option(name="--iter",usage="Number of iterations",aliases="-i")
    protected int iterations = 2;
    @Option(name="--numLabels",usage="Number of categories",aliases="-nL")
    protected int numLabels = 4;
    @Option(name="--split",usage="Percent to split for training",aliases="-split")
    protected double split = 0.5;

    public void run(String[] args) {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        // standard vars
        int seed = 123;
        int listenerFreq = 1;
        boolean appendLabels = true;
        int numBatches = NUM_IMAGES/batchSize;
        double[] splitTrainTest = new double[]{split, 1-split};
        int nTrain = (int) (numExamples * split);
        int nTest = numExamples - nTrain;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("FaceDetection");
//        sparkConf.set("spak.executor.memory", "4g");
//        sparkConf.set("spak.driver.memory", "4g");
//        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");

        ////////////// Load Gender folders /////////////
        File mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class/*");
        List<String> labels = Arrays.asList(new String[]{"man", "woman"});
        //////////////////////////////////////////////

        ////////////// Load all 10 folders /////////////
//        String[] tags = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample").list(new FilenameFilter() {
//            @Override
//            public boolean accept(File dir, String name) {
//                return dir.isDirectory();
//            }
//        });
//        List<String> labels = Arrays.asList(tags);
        //////////////////////////////////////////////

        ////////////// Load MS Folder /////////////
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "ms_sample");
//        List<String> labels = Arrays.asList(new String[]{"liv_tyler", "michelle_obama", "aaron_carter", "al_gore"});
        //////////////////////////////////////////////

        ////////////// Load sequence into RDD /////////////
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample/*");
//        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(mainPath.toString());
//        JavaPairRDD<Text,BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
//        RecordReaderBytesFunction rrFunc = new RecordReaderBytesFunction(recordReader);
//        JavaRDD<Collection<Writable>> data = filesAsBytes.map(rrFunc);
//        JavaRDD<DataSet> fullData = data.map(new CanovaDataSetFunction(-1, NUM_LABELS, false));
//        fullData.cache();
//        JavaRDD<DataSet>[] trainTestSplit = fullData.randomSplit(splitTrainTest);
        //////////////////////////////////////////////


        ////////////// Load files to DS and parallelize - seems to load more examples /////////////
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample");

        RecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, appendLabels, labels);
        try {
            recordReader.initialize(new FileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, new Random(123)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, numLabels);
        List<DataSet> allData = new ArrayList<>(numExamples);
        while(dataIter.hasNext()){
            allData.add(dataIter.next());
        }

        Collections.shuffle(allData, new Random(seed));
        Iterator<DataSet> iter = allData.iterator();
        List<DataSet> train = new ArrayList<>(nTrain);
        List<DataSet> test = new ArrayList<>(nTest);

        int c = 0;
        int d = 0;
        while(iter.hasNext()){
            if(c++ < nTrain) train.add(iter.next());
            else if(d++ < (nTrain + nTest)) test.add(iter.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());
        //////////////////////////////////////////////

        log.info("Build model....");
        MultiLayerNetwork model = new AlexNet(HEIGHT, WIDTH, CHANNELS, numLabels, seed, iterations).init();
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, model);

        log.info("Train model....");
        for(int i = 0; i < epochs; i++){
            sparkNetwork.fitDataSet(sparkDataTrain);
        }
        sparkDataTrain.unpersist();

        log.info("Evaluate model....");
//      Alternatives if have a full set and break into trainTestSplit - need to pull 0 for train
//        Evaluation evalActual = sparkNetwork.evaluate(trainTestSplit[1].coalesce(5), labels);

        JavaRDD<DataSet> sparkDataTest = sc.parallelize(test);
        sparkDataTest.persist(StorageLevel.MEMORY_ONLY());
        Evaluation evalActual = sparkNetwork.evaluate(sparkDataTest, labels);
        log.info(evalActual.stats());

        sparkDataTest.unpersist();
//        fullData.unpersist();

        log.info("****************Example finished********************");
    }

    public static void main(String[] args) throws Exception {
        new MSRA_CFWSpark().run(args);
    }

}
