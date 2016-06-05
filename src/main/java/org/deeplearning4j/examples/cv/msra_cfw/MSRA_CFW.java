package org.deeplearning4j.examples.cv.msra_cfw;


import org.canova.api.io.labels.ParentPathLabelGenerator;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.BaseImageRecordReader;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.AlexNet;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * WORK IN PROGRESS Face Classification
 *
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

public class MSRA_CFW {
    private static final Logger log = LoggerFactory.getLogger(MSRA_CFW.class);

    // Based on small sample dataset
    public final static int HEIGHT = 100;
    public final static int WIDTH = 100; // size varies
    public final static int CHANNELS = 3;

    // Values to pass in from command line when compiled, esp running remotely
    @Option(name="--numExamples",usage="Number of examples",aliases="-nE")
    protected int numExamples = 100;
    @Option(name="--batchSize",usage="Batch size",aliases="-b")
    protected int batchSize = 50;
    @Option(name="--epochs",usage="Number of epochs",aliases="-ep")
    protected int epochs = 5;
    @Option(name="--iter",usage="Number of iterations",aliases="-i")
    protected int iterations = 1;
    @Option(name="--numLabels",usage="Number of categories",aliases="-nL")
    protected int numLabels = 2;
    @Option(name="--split",usage="Percent to split for training",aliases="-split")
    protected double split = 0.8;

    protected boolean gender = true;

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

        int seed = 123;
        int listenerFreq = 1;
        boolean appendLabels = true;
        int splitTrainNum = (int) (batchSize*split);

        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();


        // TODO setup to download and untar the example - currently needs manual download

        log.info("Load data....");
        // Two options to setup gender or 10 unique names classification
        File mainPath;
        if(gender) {
            numLabels = 2;
            mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class");
        }else{
            numLabels = 10;
            mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample"); // 10 labels
        }

        RecordReader recordReader = new BaseImageRecordReader(HEIGHT, WIDTH, CHANNELS, new ParentPathLabelGenerator()) {
            @Override
            protected boolean containsFormat(String format) {
                return super.containsFormat(format);
            }
        };
        try {
            recordReader.initialize(new LimitFileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, numExamples, numLabels, null, new Random(123)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        List<String> labels = gender? Arrays.asList(new String[]{"man", "woman"}): dataIter.getLabels();

        log.info("Build model....");
        // AlexNet is one type of model provided in model sanctuary. Building your own is also an option
        MultiLayerNetwork model = new AlexNet(HEIGHT, WIDTH, CHANNELS, numLabels, seed, iterations).init();
        model.init();

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

        model.setListeners(new ScoreIterationListener(listenerFreq));
//        model.setListeners(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(listenerFreq));
//        model.setListeners(new ScoreIterationListener(listenerFreq), paramListener);


        log.info("Train model....");
        // One epoch
        DataSet dsNext;
        while (dataIter.hasNext()) {
            dsNext = dataIter.next();
            dsNext.scale();
            trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        // More than 1 epoch just for training
        for(int i = 1; i < epochs; i++) {
            dataIter.reset();
            while (dataIter.hasNext()) {
                dsNext = dataIter.next();
                trainTest = dsNext.splitTestAndTrain(splitTrainNum, new Random(seed));
                trainInput = trainTest.getTrain();
                model.fit(trainInput);
            }
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(labels);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());


        log.info("****************Example finished********************");

    }

    public static void main(String[] args) throws Exception {
        new MSRA_CFW().run(args);
    }

}