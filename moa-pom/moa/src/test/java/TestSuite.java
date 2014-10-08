import java.io.File;
import java.net.URL;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;

import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.options.ClassOption;

import org.apache.commons.io.FileUtils;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.SMOTE;


public class TestSuite {

	private static final int MAX_NUM_INSTANCES_USED_IN_ARFF = 10000;
	private static final int SMOTE_SAMPLE_SIZE = 2000;
	private static final String SMOTE_PARAS = "-C 0 -K 5 -P 90.0 -S 1";
	private static final double DESIRED_CLASS_RATIO = 1;
	private static final boolean PERFORM_SMOTE = true;

	public static void main(String[] args) throws Exception {
		Experiment exp = new Experiment();
		exp.setSampleSize(SMOTE_SAMPLE_SIZE);
		exp.setPerformSMOTE(PERFORM_SMOTE);

		SMOTE smote = new SMOTE();
		String[] options = weka.core.Utils.splitOptions(SMOTE_PARAS);
		smote.setOptions(options);
		exp.setDesiredClassRatio(DESIRED_CLASS_RATIO);
		Experiment.setSmote(smote);

		URL resource = exp.getClass().getClassLoader().getResource(".");
		String fileString = resource.getPath();
		Collection<File> files = FileUtils.listFiles(new File(fileString), new String[] { "arff" }, false);
		
		Map<String, DriftDetectionMethodClassifier> map = initializeDriftLearners();
		
		for (File file : files) {
			String currentArffAbsolutePath = file.getAbsolutePath();
			String currentFileName = file.getName();

//			System.out.println(currentArffAbsolutePath);
			exp.setCurrentArffAbsolutePath(currentArffAbsolutePath);
			DataSource source = new DataSource(currentArffAbsolutePath);
			exp.setData(source.getDataSet());
			
			for (Entry<String, DriftDetectionMethodClassifier> entry : map.entrySet()) {
				String csvFileName = currentFileName+"_"+entry.getKey();
				System.out.println(csvFileName);
				exp.setDriftLearner(entry.getValue());
				exp.run(MAX_NUM_INSTANCES_USED_IN_ARFF, true, csvFileName);
				
			}
		}

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
