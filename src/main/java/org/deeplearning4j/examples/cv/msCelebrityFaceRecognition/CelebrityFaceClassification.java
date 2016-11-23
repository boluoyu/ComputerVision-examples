package org.deeplearning4j.examples.cv.msCelebrityFaceRecognition;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.TestModels.AlexNet;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/***
 * Microsoft Research MSRA-CFW Celebrity Faces Dataset
 *
 * This is an image classification example built from scratch. You can swap out your own dataset with this structure.
 * Note additional work is needed to build out the structure to work with your dataset.
 * Dataset:
 *      - Celebrity Faces created by MicrosoftResearch
 *      - Based on thumbnails data set which is a smaller subset
 *      - 2215 images & 10 classifications with each image only including one face
 *      - Dataset has more examples of a each person than LFW to make standard classification approaches appropriate
 *      - Gender variation is something built separate from the dataset
 *
 * Checkout this link for more information and to access data: http://research.microsoft.com/en-us/projects/msra-cfw/
 */

public class CelebrityFaceClassification {
    private static final Logger log = LoggerFactory.getLogger(CelebrityFaceClassification.class);

    // Based on small sample dataset
    public final static int height = 100;
    public final static int width = 100; // size varies
    public final static int channels = 3;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;

    // Values to pass in from command line when compiled, esp running remotely
    @Option(name="--numExamples",usage="Number of examples",aliases="-nE")
    protected int numExamples = 1000;
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 200;
    @Option(name="--epochs",usage="Number of epochs",aliases="-ep")
    protected int epochs = 5;
    @Option(name="--iter",usage="Number of iterations",aliases="-i")
    protected int iterations = 1;
    @Option(name="--split",usage="Percent to split for training",aliases="-split")
    protected double splitTrainTest = 0.8;

    public void run(String[] args) throws Exception{

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        // TODO setup to download and untar the example - currently needs manual download

        log.info("Load data....");
        // Two options to setup gender or 10 unique names classification
        File mainPath;
        int numLabels;
        boolean gender = false;
        if(gender) {
            numLabels = 2;
            mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class");
        }else{
            numLabels = 10;
//            mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample"); // 10 labels
            mainPath = new File(BaseImageLoader.BASE_DIR, "data/mrsa-cfw"); // 10 labels
        }
        // Organize  & limit data file paths
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), new ParentPathLabelGenerator(), numExamples, numLabels, batchSize);

        // Setup train test split
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, numExamples*(1+splitTrainTest),  numExamples*(1-splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];


        // Define image transformations to increase dataset
        ImageTransform flipTransform = new FlipImageTransform(90);
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[] {null, flipTransform, warpTransform});


        // Define how data will load into net - use below if no transforms
//        RecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
//        recordReader.initialize(trainData);
//        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
//        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);

        log.info("Build model....");
        // AlexNet is one type of model provided in model sanctuary. Building your own is also an option
        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();
        network.init();

        // Set listeners
        // Use paramListener if only have access to command line to track performance
        IterationListener paramListener = ParamAndGradientIterationListener.builder()
                .outputToFile(true)
                .file(new File(System.getProperty("java.io.tmpdir") + "/paramAndGradTest.txt"))
                .outputToConsole(true).outputToLogger(false)
                .iterations(listenerFreq).printHeader(true)
                .printMean(false)
                .printMinMax(false)
                .printMeanAbsValue(true)
                .delimiter("\t").build();

        network.setListeners(new ScoreIterationListener(listenerFreq));
//        network.setListeners(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq));
//        network.setListeners(new ScoreIterationListener(listenerFreq), paramListener);

        log.info("Train model....");
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;

        // Train with transformations
        for(ImageTransform transform: transforms) {
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            trainIter = new MultipleEpochsIterator(epochs, dataIter);
            network.fit(trainIter);
        }

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats());


        log.info("****************Example finished********************");

    }

    public static void main(String[] args) throws Exception {
        new CelebrityFaceClassification().run(args);
    }

}