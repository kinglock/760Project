import java.io.File;
import java.net.URL;
import java.util.Collection;

import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.core.TimingUtils;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;

import org.apache.commons.io.FileUtils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class Experiment {

	private static Instances data;

	private static SMOTE smote;

	// private Classifier learner = new HoeffdingAdaptiveTree();
	private Classifier learner;
	private int numberSamplesCorrect;

	private int sampleSize = 2000;
	private static String currentArffAbsolutePath;

	public Experiment() {
	}

	public void run(int numInstances, boolean isTesting) throws Exception {
		// RandomRBFGenerator stream = new RandomRBFGenerator();
		// SEAGenerator stream = new SEAGenerator();
		ArffFileStream stream = new ArffFileStream(currentArffAbsolutePath, -1);
		stream.prepareForUse();

		learner.setModelContext(stream.getHeader());
		learner.prepareForUse();

		int numberSamples = 0;
		boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

		/*
		 * Generate a fixed size sample
		 */
		int curentSize = 0;
		Instances sample = new Instances(data, sampleSize);

		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			if (curentSize < sampleSize) {

				Instance trainInst = stream.nextInstance();

				numberSamples++;
				sample.add(trainInst);
				curentSize++;
			} else {
				Instances newDataset = applySMOTE(sample);
				training(newDataset);
				testPerformance(sample, numberSamples);
				curentSize = 0;
				sample.clear();
			}

		}
		/*
		 * Last sample if it still has some instances in it
		 */
		if (!sample.isEmpty()) {
			Instances newDataset = applySMOTE(sample);
			training(newDataset);
			testPerformance(sample, numberSamples);
			curentSize = 0;
			sample.clear();

		}

		/*
		 * Using trained model for testing the entire stream
		 */
		int count = 0;
		int count_class_zero = 0;
		if (isTesting) {
			numberSamplesCorrect = 0;
			stream = new ArffFileStream(currentArffAbsolutePath, -1);
			stream.prepareForUse();
			while (stream.hasMoreInstances()) {
				Instance testInst = stream.nextInstance();
				if (learner.correctlyClassifies(testInst)) {
					numberSamplesCorrect++;
				}
				if (testInst.classValue() == 0) {
					count_class_zero++;
				}
				count++;
			}
		}

		double accuracy = 100.0 * (double) numberSamplesCorrect / (double) count;
		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
		System.out.println(count + " instances processed with " + accuracy + "% accuracy in " + time
				+ " seconds.");
		System.out.println(count_class_zero + " instances with class value of 0!");
	}

	private void testPerformance(Instances sample, int numberSamples) {
		numberSamplesCorrect = 0;
		int index = 0;
		while (index < numberSamples) {
			if (learner.correctlyClassifies(data.instance(index))) {
				numberSamplesCorrect++;
			}
			index++;
		}

		double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
		System.err.println(numberSamples + " Current accuracy is: " + accuracy + "%");

	}

	private Instances applySMOTE(Instances sample) throws Exception {

		// System.out.println("starting applying smote...");
		// System.out.println(sample.attributeStats(sample.numAttributes() -
		// 1));

		/*
		 * ignore if there is only one class
		 */
		int distinctCount = sample.attributeStats(sample.classIndex()).distinctCount;
		if (distinctCount == 1) {
			return sample;
		}

		smote.setInputFormat(sample);
		Instances newDataset = Filter.useFilter(sample, smote);
		// System.out.println("after applying SMOTE:"+newDataset.toSummaryString());
		System.out.println(newDataset.attributeStats(newDataset.numAttributes() - 1));
		return newDataset;
	}

	private void training(Instances newDataset) {
		for (Instance trainInst : newDataset) {
			learner.trainOnInstance(trainInst);
		}
	}

	// public static void main(String[] args) throws Exception {
	//
	// smote = new SMOTE();
	// String[] options =
	// weka.core.Utils.splitOptions("-C 0 -K 5 -P 90.0 -S 1");
	// smote.setOptions(options);
	//
	// Experiment exp = new Experiment();
	// URL resource = exp.getClass().getClassLoader().getResource(".");
	// String fileString = resource.getPath();
	// System.out.println(fileString);
	// Collection<File> files = FileUtils.listFiles(new File(fileString), new
	// String[] { "arff" }, false);
	// for (File file : files) {
	// currentArffAbsolutePath = file.getAbsolutePath();
	// System.out.println(currentArffAbsolutePath);
	// DataSource source = new DataSource(currentArffAbsolutePath);
	// data = source.getDataSet();
	// if (data.classIndex() == -1)
	// data.setClassIndex(data.numAttributes() - 1);
	//
	// exp.run(1000000, true);
	// }
	//
	// }

	public static void setSmote(SMOTE smote) {
		Experiment.smote = smote;
	}

	public void setCurrentArffAbsolutePath(String currentArffAbsolutePath2) {
		Experiment.currentArffAbsolutePath = currentArffAbsolutePath2;

	}

	public void setData(Instances dataSet) {
		Experiment.data = dataSet;
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);

	}

	public void setDriftLearner(DriftDetectionMethodClassifier driftDetectionMethodClassifier) {

		learner = driftDetectionMethodClassifier;
	}

	public void setSampleSize(int sampleSize) {
		this.sampleSize = sampleSize;
	}
}