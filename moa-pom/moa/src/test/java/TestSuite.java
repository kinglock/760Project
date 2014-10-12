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
	private static final double[] IMBALANCE_RATIO_IN_STREAM = {0.01,0.1};
	private static final int POSITION = 4000; // position of abrupt drift 
	private static final int WIDTH = 5000; // width of abrupt drift

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
			for(int seed = 1; seed <= NUM_OF_SEEDS; seed++){
				String currentFileName = entryOfOneDataset.getKey()+"_SEED_"+(seed);
				InstanceStream stream = streamsOfOneDataset.get(seed-1);
				System.out.println("current loop seed is "+seed);
				try {
					ConceptDriftStream stream1;
					stream1 = (ConceptDriftStream) stream;
					System.out.println("current stream seed is "+stream1.randomSeedOption.getValue());
				} catch (ClassCastException e) {
					StaggerImbalanced stream1 = (StaggerImbalanced) stream;
					stream1 = (StaggerImbalanced) stream;
					System.out.println("current stream seed is "+stream1.instanceRandomSeedOption.getValue());
				}
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
		List<InstanceStream> streamOfOneDatasetWithDifferentSeed1= new LinkedList<InstanceStream>();
		List<InstanceStream> streamOfOneDatasetWithDifferentSeed2 = new LinkedList<InstanceStream>();
		
		for (double imblanceRatio : IMBALANCE_RATIO_IN_STREAM) {
			for(int seed = 1; seed <= NUM_OF_SEEDS; seed++){
				InstanceStream stream1 = Experiment.createImbalancedStaggerDriftStream(POSITION, WIDTH, imblanceRatio, seed, null);
				streamOfOneDatasetWithDifferentSeed1.add(stream1);
				InstanceStream stream2 = Experiment.createImbalancedStaggerNoDriftStream(imblanceRatio, seed);
				streamOfOneDatasetWithDifferentSeed2.add(stream2);
			}
			map.put("StaggerAbruptDrift_Imbalanced_"+imblanceRatio, streamOfOneDatasetWithDifferentSeed1);
			map.put("StaggerNoDrift_Imbalanced_"+imblanceRatio, streamOfOneDatasetWithDifferentSeed2);
			
		}

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
				Classifier.class, "trees.HoeffdingTree"); // why are we using Adaptive HT here??
		// shouldn't we use the same base classifier HT for both methods?
		driftDetectionMethodClassifier2.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
				"Drift detection method to use.", ChangeDetector.class, "PageHinkleyDM "+PHT_PARAS);
		driftDetectionMethodClassifier2.driftDetectionMethodOption.getValueAsCLIString();
		
		
		map.put("HoeffdingTree_ADWINChangeDetector",driftDetectionMethodClassifier1);
		map.put("HoeffdingTree_PageHinkleyDM",driftDetectionMethodClassifier2);
		
		
		return map;
		
	}


}
