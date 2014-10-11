
import datastream.streams.StreamGen;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import weka.filters.supervised.instance.SMOTE;

public class TestSuite {

    private static final int NUM_OF_SEEDS = 1;
    private static final int MAX_NUM_INSTANCES_USED_IN_ARFF = 1000000;
    private static final int SMOTE_SAMPLE_SIZE = 2000;
    private static final String SMOTE_PARAS = "-C 0 -K 5 -P 90.0 -S 1";
    private static final double DESIRED_CLASS_RATIO = 1;
    private static final boolean PERFORM_SMOTE = false;
    private static final double[] IMBALANCE_RATIO_IN_STREAM = {0.01, 0.1, 0.5};

    private static final int POSITION = 300000; // position of abrupt drift 
    private static final int WIDTH = 50000; // width of abrupt drift (use this for less abrupt sigmoidal drifts)
    private static final int ALPHA = 90; // angle of abrupt drift (use this for more abrupt drifts) 
    private static final boolean USE_WIDTH = false; // use either width or angle 

    private static final double SPEED = 0.01; // speed of gradual drift
    private static final int CENTROIDS = 3; // number of centroids with drift for gruadual drift stream  

    public static void main(String[] args) throws Exception {

        Experiment exp = new Experiment();
        exp.setSampleSize(SMOTE_SAMPLE_SIZE);
        exp.setPerformSMOTE(PERFORM_SMOTE);

        SMOTE smote = new SMOTE();
        String[] options = weka.core.Utils.splitOptions(SMOTE_PARAS);
        smote.setOptions(options);
        exp.setDesiredClassRatio(DESIRED_CLASS_RATIO);
        Experiment.setSmote(smote);

        Map<String, DriftDetectionMethodClassifier> map = initializeDriftLearners();
        Map<String, List<InstanceStream>> streams = initializeGenerator();

        for (Entry<String, List<InstanceStream>> entryOfOneDataset : streams.entrySet()) {
            List<InstanceStream> streamsOfOneDataset = entryOfOneDataset.getValue();
            for (int seed = 1; seed <= streamsOfOneDataset.size(); seed++) {
                String currentFileName = entryOfOneDataset.getKey() + "_SEED_" + (seed);
                InstanceStream stream = streamsOfOneDataset.get(seed - 1);
                exp.setStream(stream);
                exp.setTestStream(stream);
                for (Entry<String, DriftDetectionMethodClassifier> entry : map.entrySet()) {
                    String csvFileName = currentFileName + "_" + entry.getKey();
                    exp.setDriftLearner(entry.getValue());
                    exp.run(MAX_NUM_INSTANCES_USED_IN_ARFF, true, csvFileName);
                }
            }
        }

    }

    private static Map<String, List<InstanceStream>> initializeGenerator() throws IOException {
        Map<String, List<InstanceStream>> map = new HashMap<>();

        List<InstanceStream> streamOfOneDataset1 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset2 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset3 = new LinkedList<>();

        List<InstanceStream> streamOfOneDataset4 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset5 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset6 = new LinkedList<>();

        List<InstanceStream> streamOfOneDataset7 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset8 = new LinkedList<>();
        List<InstanceStream> streamOfOneDataset9 = new LinkedList<>();

        for (int seed = 1; seed <= NUM_OF_SEEDS; seed++) {

            // createImbalancedStaggerDriftStream(double imbalance, int pos, int width, double alpha, boolean useWidth, int seed)
            InstanceStream stream1 = StreamGen.createImbalancedStaggerDriftStream(IMBALANCE_RATIO_IN_STREAM[0], POSITION, WIDTH, ALPHA, USE_WIDTH, seed);
            streamOfOneDataset1.add(stream1);
            InstanceStream stream2 = StreamGen.createImbalancedStaggerDriftStream(IMBALANCE_RATIO_IN_STREAM[1], POSITION, WIDTH, ALPHA, USE_WIDTH, seed);
            streamOfOneDataset2.add(stream2);
            InstanceStream stream3 = StreamGen.createImbalancedStaggerDriftStream(IMBALANCE_RATIO_IN_STREAM[2], POSITION, WIDTH, ALPHA, USE_WIDTH, seed);
            streamOfOneDataset3.add(stream3);

            // createImbalancedStaggerNoDriftStream(double imbalance, int f, int seed)
            InstanceStream stream4 = StreamGen.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[0], 1, seed);
            streamOfOneDataset4.add(stream4);
            InstanceStream stream5 = StreamGen.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[1], 1, seed);
            streamOfOneDataset5.add(stream5);
            InstanceStream stream6 = StreamGen.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[2], 1, seed);
            streamOfOneDataset6.add(stream6);

            // createRBFDriftStream(double imbalance, int centroids, double speed, int seed) 
            InstanceStream stream7 = StreamGen.createRBFDriftStream(IMBALANCE_RATIO_IN_STREAM[0], CENTROIDS, SPEED, seed);
            streamOfOneDataset7.add(stream7);
            InstanceStream stream8 = StreamGen.createRBFDriftStream(IMBALANCE_RATIO_IN_STREAM[1], CENTROIDS, SPEED, seed);
            streamOfOneDataset8.add(stream8);
            InstanceStream stream9 = StreamGen.createRBFDriftStream(IMBALANCE_RATIO_IN_STREAM[2], CENTROIDS, SPEED, seed);
            streamOfOneDataset9.add(stream9);
        }

        map.put("StaggerAbruptDrift_Imbalanced_A", streamOfOneDataset1);
        map.put("StaggerAbruptDrift_Imbalanced_B", streamOfOneDataset2);
        map.put("StaggerAbruptDrift_Imbalanced_C", streamOfOneDataset3);

        map.put("StaggerNoDrift_Imbalanced_A", streamOfOneDataset4);
        map.put("StaggerNoDrift_Imbalanced_B", streamOfOneDataset5);
        map.put("StaggerNoDrift_Imbalanced_C", streamOfOneDataset6);

        map.put("RBFGradualDrift_Imbalanced_A", streamOfOneDataset7);
        map.put("RBFGradualDrift_Imbalanced_B", streamOfOneDataset8);
        map.put("RBFGradualDrift_Imbalanced_C", streamOfOneDataset9);

        return map;

    }

    private static Map<String, DriftDetectionMethodClassifier> initializeDriftLearners() {
        Map<String, DriftDetectionMethodClassifier> map = new HashMap<>();

        DriftDetectionMethodClassifier driftDetectionMethodClassifier1 = new DriftDetectionMethodClassifier();
        driftDetectionMethodClassifier1.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
                Classifier.class, "trees.HoeffdingTree");
        driftDetectionMethodClassifier1.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
                "Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");

        DriftDetectionMethodClassifier driftDetectionMethodClassifier2 = new DriftDetectionMethodClassifier();
        driftDetectionMethodClassifier2.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
                Classifier.class, "trees.HoeffdingTree"); // modified: now using HT rather than Adaptive HT        
        driftDetectionMethodClassifier2.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
                "Drift detection method to use.", ChangeDetector.class, "PageHinkleyDM");

        map.put("HoeffdingTree_ADWINChangeDetector", driftDetectionMethodClassifier1);
        map.put("HoeffdingTree_PageHinkleyDM", driftDetectionMethodClassifier2);

        return map;

    }

}
