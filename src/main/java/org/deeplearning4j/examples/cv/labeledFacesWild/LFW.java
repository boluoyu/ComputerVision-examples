package org.deeplearning4j.examples.cv.labeledFacesWild;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.cv.TestModels.DeepFaceVariant;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Random;

/**
 * Labeled Faces in the Wild
 *
 * Dataset created by Erik Learned-Miller, Gary Huang, Aruni RoyChowdhury,
 * Haoxiang Li, Gang Hua. This is used to study unconstrained face recognition.
 * Each face has been labeled with the name of the person pictured.
 *
 * Over 13K images
 * 5749 unique classes (different people)
 * 1680 people have 2+ photos
 *
 * References:
 * General information is at http://vis-www.cs.umass.edu/lfw/.
 *
 * Note: this is a sparse dataset with only 1 example for many of the faces; thus, performance is low.
 * Ideally train on a larger dataset like celebs to get params and/or generate variations of the image examples.
 *
 * Currently set to only use the subset images, names starting with A.
 * Switch to NUM_LABELS & NUM_IMAGES and set subset to false to use full dataset.
 */

public class LFW {
    private static final Logger log = LoggerFactory.getLogger(LFW.class);

    protected static final int height = 100;
    protected static final int width = 100;
    protected static final int channels = 3;

    protected static int numExamples = LFWLoader.NUM_IMAGES;
    protected static int numLabels = LFWLoader.SUB_NUM_LABELS;
    protected static int batchSize = 128;
    protected static boolean useSubset = true;
    protected static int seed = 42;
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 5;
    protected static double splitTrainTest = 0.8;
    protected static int normalizeVal = 255;
    protected static int nCores = 32;

    public static void main(String[] args) {

        log.info("Load data training data....");
        LFWDataSetIterator lfw = new LFWDataSetIterator(batchSize, numExamples, new int[] {height, width, channels}, numLabels, useSubset, new ParentPathLabelGenerator(), true, splitTrainTest, null, normalizeVal, new Random(seed));

        log.info("Build model....");
        MultiLayerNetwork network = new DeepFaceVariant(height, width, channels, numLabels, seed, iterations).init();
//        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        log.info("Train model....");
        network.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        MultipleEpochsIterator multiLFW = new MultipleEpochsIterator(epochs, lfw, nCores);
        network.fit(multiLFW);

        log.info("Load data testing data....");
        lfw = new LFWDataSetIterator(batchSize, numExamples, new int[] {height, width, channels}, numLabels, useSubset, new ParentPathLabelGenerator(), false, splitTrainTest, null, normalizeVal, new Random(seed));


        log.info("Evaluate model....");
        Evaluation eval = network.evaluate(lfw, lfw.getLabels());
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
