import java.io.File;
import java.net.URL;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

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

	public static void main(String[] args) throws Exception {
		Experiment exp = new Experiment();
		exp.setSampleSize(SMOTE_SAMPLE_SIZE);

		SMOTE smote = new SMOTE();
		String[] options = weka.core.Utils.splitOptions(SMOTE_PARAS);
		smote.setOptions(options);
		Experiment.setSmote(smote);

		URL resource = exp.getClass().getClassLoader().getResource(".");
		String fileString = resource.getPath();
		Collection<File> files = FileUtils.listFiles(new File(fileString), new String[] { "arff" }, false);
		
		List<DriftDetectionMethodClassifier> initializeDriftLearner = initializeDriftLearners();
		
		for (File file : files) {
			String currentArffAbsolutePath = file.getAbsolutePath();
			System.out.println(currentArffAbsolutePath);
			exp.setCurrentArffAbsolutePath(currentArffAbsolutePath);
			DataSource source = new DataSource(currentArffAbsolutePath);
			exp.setData(source.getDataSet());
			
			for (DriftDetectionMethodClassifier driftDetectionMethodClassifier : initializeDriftLearner) {
				exp.setDriftLearner(driftDetectionMethodClassifier);
				
				exp.run(MAX_NUM_INSTANCES_USED_IN_ARFF, true);
			}

		}

	}

	private static List<DriftDetectionMethodClassifier> initializeDriftLearners() {
		List<DriftDetectionMethodClassifier> list = new LinkedList<DriftDetectionMethodClassifier>();
		
		DriftDetectionMethodClassifier driftDetectionMethodClassifier1 = new DriftDetectionMethodClassifier();
		driftDetectionMethodClassifier1.driftDetectionMethodOption = new ClassOption("baseLearner", 'l', "Classifier to train.",
				Classifier.class, "trees.HoeffdingTree");
		driftDetectionMethodClassifier1.driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
				"Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");
		
		
		list.add(driftDetectionMethodClassifier1);
		
		
		return list;
		
	}


}
