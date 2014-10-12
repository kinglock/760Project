
import datastream.streams.RBFDrift;
import datastream.streams.StaggerImbalanced;
import datastream.streams.StreamGen;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.SEAGenerator;
import weka.filters.supervised.instance.SMOTE;

public class TestSuite {

    private static final String PHT_PARAS = "-n 15 -d 0.015";
    private static final int NUM_OF_SEEDS = 30;
    private static final int MAX_NUM_INSTANCES_USED_IN_ARFF = 1000000;
    private static final int SMOTE_SAMPLE_SIZE = 2000;
    private static final String SMOTE_PARAS = "-C 0 -K 5 -P 90.0 -S 1";
    private static final double DESIRED_CLASS_RATIO = 1;

    private static final boolean PERFORM_SMOTE = true;
    // private static final double[] IMBALANCE_RATIO_IN_STREAM = {0.01, 0.1, 0.5};
    private static final double[] IMBALANCE_RATIO_IN_STREAM = {0.01};

    private static final int POSITION = 300000; // position of abrupt drift
    private static final int WIDTH = 50000; // width of abrupt drift    
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
            for (int seed = 1; seed <= NUM_OF_SEEDS; seed++) {
                String currentFileName = entryOfOneDataset.getKey() + "_SEED_" + (seed);
                InstanceStream stream = streamsOfOneDataset.get(seed - 1);
                System.out.println("current loop seed is " + seed);
                try {
                    ConceptDriftStream stream1 = (ConceptDriftStream) stream;
                    System.out.println("current stream seed is " + stream1.randomSeedOption.getValue());
                } catch (ClassCastException ex1) {
                    try {
                        StaggerImbalanced stream1 = (StaggerImbalanced) stream;
                        System.out.println("current stream seed is " + stream1.instanceRandomSeedOption.getValue());
                    } catch (ClassCastException ex2) {
                        RBFDrift stream1 = (RBFDrift) stream; // added third stream type
                        System.out.println("current stream seed is " + stream1.instanceRandomSeedOption.getValue());
                    }
                }
                exp.setStream(stream);
                exp.setTestStream(stream);
                for (Entry<String, DriftDetectionMethodClassifier> entry : map.entrySet()) {
                    String learnerName = entry.getKey();
                    String csvFileName = currentFileName + "_" + learnerName;
                    // need to create new learner each time as a work around for setting different contexts (setModelContext)
                    DriftDetectionMethodClassifier learner = new DriftDetectionMethodClassifier();
                    if (learnerName.equals("HoeffdingTree_ADWINChangeDetector")) {
                        learner = createADWIN();
                    } else if (learnerName.equals("HoeffdingTree_PageHinkleyDM")) {
                        learner = createPHT();
                    }
                    exp.setDriftLearner(learner);
                    exp.run(MAX_NUM_INSTANCES_USED_IN_ARFF, true, csvFileName);

                }
            }
        }
    }

    private static Map<String, List<InstanceStream>> initializeGenerator() {
        Map<String, List<InstanceStream>> map = new HashMap<String, List<InstanceStream>>();
        List<InstanceStream> streamOfOneDatasetWithDifferentSeed1 = new LinkedList<InstanceStream>();
        List<InstanceStream> streamOfOneDatasetWithDifferentSeed2 = new LinkedList<InstanceStream>();
        List<InstanceStream> streamOfOneDatasetWithDifferentSeed3 = new LinkedList<InstanceStream>();
        for (double imblanceRatio : IMBALANCE_RATIO_IN_STREAM) {
            for (int seed = 1; seed <= NUM_OF_SEEDS; seed++) {
                InstanceStream stream1 = StreamGen.createImbalancedStaggerDriftStream(imblanceRatio, POSITION, WIDTH, ALPHA, USE_WIDTH, seed);
                streamOfOneDatasetWithDifferentSeed1.add(stream1);
                InstanceStream stream2 = StreamGen.createImbalancedStaggerNoDriftStream(imblanceRatio, 1, seed);
                streamOfOneDatasetWithDifferentSeed2.add(stream2);
                InstanceStream stream3 = StreamGen.createRBFDriftStream(imblanceRatio, CENTROIDS, SPEED, seed);
                streamOfOneDatasetWithDifferentSeed3.add(stream3);
            }
            map.put("StaggerAbruptDrift_Imbalanced_" + imblanceRatio, streamOfOneDatasetWithDifferentSeed1); // abrupt drift
            map.put("StaggerNoDrift_Imbalanced_" + imblanceRatio, streamOfOneDatasetWithDifferentSeed2); // no drift
            map.put("RBFGradualDrift_Imbalanced_" + imblanceRatio, streamOfOneDatasetWithDifferentSeed3); // gradual drift
        }
        return map;
    }

    private static Map<String, DriftDetectionMethodClassifier> initializeDriftLearners() {
        Map<String, DriftDetectionMethodClassifier> map = new HashMap<String, DriftDetectionMethodClassifier>();

        DriftDetectionMethodClassifier driftDetectionMethodClassifier1 = createADWIN(); // refactored
        DriftDetectionMethodClassifier driftDetectionMethodClassifier2 = createPHT();

        driftDetectionMethodClassifier2.driftDetectionMethodOption.getValueAsCLIString();

        map.put("HoeffdingTree_ADWINChangeDetector", driftDetectionMethodClassifier1);
        map.put("HoeffdingTree_PageHinkleyDM", driftDetectionMethodClassifier2);
        return map;
    }

    // refactored creation of classifiers into separate methods
    private static DriftDetectionMethodClassifier createADWIN() {
        DriftDetectionMethodClassifier learner = new DriftDetectionMethodClassifier();
        learner.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
                Classifier.class, "trees.HoeffdingTree");
        learner.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
                "Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");
        return learner;
    }

    private static DriftDetectionMethodClassifier createPHT() {
        DriftDetectionMethodClassifier learner = new DriftDetectionMethodClassifier();
        learner.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
                Classifier.class, "trees.HoeffdingTree"); // modified: now using HT rather than Adaptive HT          
        learner.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
                "Drift detection method to use.", ChangeDetector.class, "PageHinkleyDM " + PHT_PARAS); // using PHT options
        return learner;
    }
}
