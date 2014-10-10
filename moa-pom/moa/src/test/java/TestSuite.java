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
import moa.streams.InstanceStream;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.SEAGenerator;
import weka.filters.supervised.instance.SMOTE;


public class TestSuite {

	private static final int NUM_OF_SEEDS = 2;
	private static final int MAX_NUM_INSTANCES_USED_IN_ARFF = 10000;
	private static final int SMOTE_SAMPLE_SIZE = 2000;
	private static final String SMOTE_PARAS = "-C 0 -K 5 -P 90.0 -S 1";
	private static final double DESIRED_CLASS_RATIO = 1;
	private static final boolean PERFORM_SMOTE = true;
	private static final double[] IMBALANCE_RATIO_IN_STREAM = {0.01, 0.1, 0.5};
	private static final int POSITION = 4000;
	private static final int WIDTH = 5000;

	public static void main(String[] args) throws Exception {
		System.out.println("--------------------------------");
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
			for(int seed = 1; seed <= streamsOfOneDataset.size(); seed++){
				String currentFileName = entryOfOneDataset.getKey()+"_SEED_"+(seed);
				InstanceStream stream = streamsOfOneDataset.get(seed-1);
				exp.setStream(stream);
				exp.setTestStream(stream);
				for (Entry<String, DriftDetectionMethodClassifier> entry : map.entrySet()) {
					String csvFileName = currentFileName+"_"+entry.getKey();
					exp.setDriftLearner(entry.getValue());
					exp.run(MAX_NUM_INSTANCES_USED_IN_ARFF, true, csvFileName);
					
				}
			}

		}

	}

	private static Map<String, List<InstanceStream>> initializeGenerator() {
		Map<String, List<InstanceStream>> map = new HashMap<String, List<InstanceStream>>();
		
		List<InstanceStream> streamOfOneDataset1 = new LinkedList<InstanceStream>();
		List<InstanceStream> streamOfOneDataset2 = new LinkedList<InstanceStream>();
		List<InstanceStream> streamOfOneDataset3 = new LinkedList<InstanceStream>();
		
		List<InstanceStream> streamOfOneDataset4 = new LinkedList<InstanceStream>();
		List<InstanceStream> streamOfOneDataset5 = new LinkedList<InstanceStream>();
		List<InstanceStream> streamOfOneDataset6 = new LinkedList<InstanceStream>();
		for(int seed = 1; seed <= NUM_OF_SEEDS; seed++){
//			RandomRBFGenerator stream1 = new RandomRBFGenerator();
//			stream1.modelRandomSeedOption = new IntOption("modelRandomSeed",
//		            'r', "Seed for random generation of model.", seed);
//			streamOfOneDataset1.add(stream1);
//			
//			SEAGenerator stream2 = new SEAGenerator();
//			stream2.instanceRandomSeedOption = new IntOption(
//		            "instanceRandomSeed", 'i',
//		            "Seed for random generation of instances.", seed);
//			streamOfOneDataset2.add(stream2);	
			
			InstanceStream stream1 = Experiment.createImbalancedStaggerDriftStream(POSITION, WIDTH, IMBALANCE_RATIO_IN_STREAM[0], seed, null);
			streamOfOneDataset1.add(stream1);
			InstanceStream stream2 = Experiment.createImbalancedStaggerDriftStream(POSITION, WIDTH, IMBALANCE_RATIO_IN_STREAM[1], seed, null);
			streamOfOneDataset2.add(stream2);
			InstanceStream stream3 = Experiment.createImbalancedStaggerDriftStream(POSITION, WIDTH, IMBALANCE_RATIO_IN_STREAM[2], seed, null);
			streamOfOneDataset3.add(stream3);
			
			InstanceStream stream4 = Experiment.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[0], seed);
			streamOfOneDataset4.add(stream4);
			InstanceStream stream5 = Experiment.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[1], seed);
			streamOfOneDataset5.add(stream5);
			InstanceStream stream6 = Experiment.createImbalancedStaggerNoDriftStream(IMBALANCE_RATIO_IN_STREAM[2], seed);
			streamOfOneDataset6.add(stream6);
		}
		
		map.put("StaggerAbruptDrift_Imbalanced_A", streamOfOneDataset1);
		map.put("StaggerAbruptDrift_Imbalanced_B", streamOfOneDataset2);
		map.put("StaggerAbruptDrift_Imbalanced_C", streamOfOneDataset3);
		
		map.put("StaggerNoDrift_Imbalanced_A", streamOfOneDataset4);
		map.put("StaggerNoDrift_Imbalanced_B", streamOfOneDataset5);
		map.put("StaggerNoDrift_Imbalanced_C", streamOfOneDataset6);

		return map;
		
	}

	private static Map<String, DriftDetectionMethodClassifier> initializeDriftLearners() {
		Map<String, DriftDetectionMethodClassifier> map = new HashMap<String, DriftDetectionMethodClassifier>();
		
		DriftDetectionMethodClassifier driftDetectionMethodClassifier1 = new DriftDetectionMethodClassifier();
		driftDetectionMethodClassifier1.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
				Classifier.class, "trees.HoeffdingTree");
		driftDetectionMethodClassifier1.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
				"Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");
		
		DriftDetectionMethodClassifier driftDetectionMethodClassifier2 = new DriftDetectionMethodClassifier();
		driftDetectionMethodClassifier2.baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
				Classifier.class, "trees.HoeffdingAdaptiveTree");
		driftDetectionMethodClassifier2.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
				"Drift detection method to use.", ChangeDetector.class, "PageHinkleyDM");
		
		
		map.put("HoeffdingTree_ADWINChangeDetector",driftDetectionMethodClassifier1);
		map.put("HoeffdingAdaptiveTree_PageHinkleyDM",driftDetectionMethodClassifier2);
		
		
		return map;
		
	}


}
