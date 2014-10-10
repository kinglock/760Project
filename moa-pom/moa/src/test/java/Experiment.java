import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import moa.classifiers.Classifier;
import moa.classifiers.drift.DriftDetectionMethodClassifier;
import moa.core.TimingUtils;
import moa.options.OptionHandler;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.tasks.WriteStreamToARFFFile;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class Experiment {

	private static final int CLASS_INDEX = 1;

	private double desiredClassRatio;

	private static Instances data;

	private static SMOTE smote;

	// private Classifier learner = new HoeffdingAdaptiveTree();
	private Classifier learner;
	private int numberSamplesCorrect;

	private int sampleSize = 2000;

	private boolean performSmote = true;
	
	private static String currentArffAbsolutePath;

	private InstanceStream stream;
	private InstanceStream testStream;
	private BufferedWriter bw;

	public Experiment() {
	}

	public void run(int numInstances, boolean isTesting, String csvFileName) throws Exception {
		bw = new BufferedWriter(new FileWriter(csvFileName + ".csv"));
//		stream = new ArffFileStream(currentArffAbsolutePath, -1);
		((OptionHandler) this.stream).prepareForUse();
		((OptionHandler) this.testStream).prepareForUse();

		learner.setModelContext(this.stream.getHeader());
		learner.prepareForUse();

		int numberSamples = 0;
		boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
		long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

		/*
		 * Generate a fixed size sample
		 */
		int curentSize = 0;
		Instances sample = new Instances(this.stream.getHeader(), sampleSize);

		while (this.stream.hasMoreInstances() && numberSamples < numInstances) {
			if (curentSize < sampleSize) {

				Instance trainInst = this.stream.nextInstance();

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
		int tP = 0, tN = 0, fP = 0, fN = 0;
		if (isTesting) {
			numberSamplesCorrect = 0;
			this.testStream.restart();
			while (this.testStream.hasMoreInstances() && count < numInstances) {
				Instance testInst = this.testStream.nextInstance();
//				if (learner.correctlyClassifies(testInst)) {
//					numberSamplesCorrect++;
//				}
				
				if (learner.correctlyClassifies(testInst)) { // correct classification
                    numberSamplesCorrect++;
                    if (Utils.maxIndex(learner.getVotesForInstance(testInst)) == CLASS_INDEX) { // true positive
                        tP++;
                    } else { // true negative
                        tN++;
                    }
                } else { // incorrect classification
                    if (Utils.maxIndex(learner.getVotesForInstance(testInst)) == CLASS_INDEX) { // false positive
                        fP++;
                    } else { // false negative
                        fN++;
                    }
                }
				if (testInst.classValue() == 0) {
					count_class_zero++;
				}
				count++;
				
			}
		}
		
		double accuracy = calculateAccuracy(numberSamplesCorrect, numInstances);
		double recall = calculateRecall(tP, fN);
		double precision = calculatePrecision(tP, fP);
		double f1 = calculateF1(precision, recall);
		
		double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
		System.out.println(csvFileName + ": "+ + count + " instances processed with " + accuracy + "% accuracy in " + time
				+ " seconds.");
		System.out.println(count_class_zero + " instances with class value of 0!");
		
//		bw.write(count + "\t" + accuracy + "\t" + precision + "\t" + recall + "\t" + f1 + "\n");
		bw.close();
	}

	private double calculatePrecision(int tP, int fP) {
		// TODO Auto-generated method stub
		return (double) tP / (double) (tP + fP);
	}

	private double calculateRecall(int tP, int fN) {
		// TODO Auto-generated method stub
		return (double) tP / (double) (tP + fN);
	}

	private double calculateF1(double precision, double recall) {
		// TODO Auto-generated method stub
		return (double) (2 * precision * recall) / (double) (precision + recall);
	}

	private double calculateAccuracy(int numCorrect, int totalNum) {
		// TODO Auto-generated method stub
		return 100.0 * (double) (numCorrect) / (double) (totalNum);
		
	}

	private void testPerformance(Instances sample, int numberSamples) throws IOException {
		numberSamplesCorrect = 0;
		int tP = 0, tN = 0, fP = 0, fN = 0;
		int index = 0;
		this.testStream.restart();
		while (index < numberSamples && testStream.hasMoreInstances()) {
			index++;
			Instance instance = testStream.nextInstance();
			
			if (learner.correctlyClassifies(instance)) { // correct classification
                numberSamplesCorrect++;
                if (Utils.maxIndex(learner.getVotesForInstance(instance)) == CLASS_INDEX) { // true positive
                    tP++;
                } else { // true negative
                    tN++;
                }
            } else { // incorrect classification
                if (Utils.maxIndex(learner.getVotesForInstance(instance)) == CLASS_INDEX) { // false positive
                    fP++;
                } else { // false negative
                    fN++;
                }
            }
		}
		double accuracy = calculateAccuracy(numberSamplesCorrect, numberSamples);
		double recall = calculateRecall(tP, fN);
		double precision = calculatePrecision(tP, fP);
		double f1 = calculateF1(precision, recall);
		System.err.println(numberSamples + " Current accuracy is: " + accuracy + "% on test stream");
		bw.write(numberSamples + "\t" + accuracy + "\t" + precision + "\t" + recall + "\t" + f1 + "\n");


	}
	
	// no concept drift stream
	 public static InstanceStream createImbalancedStaggerNoDriftStream(double imbalance, int seed) {
	        StaggerImbalanced stagger = new StaggerImbalanced();
	        stagger.imbalanceWeightOption.setValue(imbalance);
	        stagger.imbalanceClassesOption.setValue(true);
	        stagger.functionOption.setValue(2); // use function 2
	        stagger.instanceRandomSeedOption.setValue(seed); // stream seed
	        stagger.prepareForUse();
	        return stagger;
	 }
	
	
	// abrupt concept drift stream
	 public static InstanceStream createImbalancedStaggerDriftStream(int position, int width, double imbalance, int seed, String filename) {
	        StaggerImbalanced stagger = new StaggerImbalanced();
	        stagger.imbalanceWeightOption.setValue(imbalance);
	        stagger.imbalanceClassesOption.setValue(true);
	        stagger.functionOption.setValue(2); // use function 2
	        stagger.instanceRandomSeedOption.setValue(222); // stream seed
	        stagger.prepareForUse();
	        //checkImbalance(stagger, 100000);

	        StaggerImbalanced staggerDrift = new StaggerImbalanced(); // drift stream with function 1
	        staggerDrift.imbalanceWeightOption.setValue(imbalance);
	        staggerDrift.imbalanceClassesOption.setValue(true);
	        staggerDrift.prepareForUse();
	        //checkImbalance(staggerDrift, 100000);

	        ConceptDriftStream driftStream = new ConceptDriftStream();
	        driftStream.streamOption.setCurrentObject(stagger); // combines two stagger streams        
	        driftStream.driftstreamOption.setCurrentObject(staggerDrift); // set drift stream
	        driftStream.positionOption.setValue(position);
	        driftStream.widthOption.setValue(width);
	        driftStream.randomSeedOption.setValue(seed); // seed for combining streams

	        driftStream.prepareForUse();
	        //System.out.println("streamOption " + driftStream.streamOption.getValueAsCLIString());
	        //System.out.println("driftStream " + driftStream.driftstreamOption.getValueAsCLIString());

	        if (filename != null) {
	            WriteStreamToARFFFile file = new WriteStreamToARFFFile();
	            file.streamOption.setCurrentObject(driftStream);
	            file.maxInstancesOption.setValue(1000000);
	            file.arffFileOption.setValue(filename);
	            file.doTask();
	        }
	        return driftStream;
	    }	 	
	 
	 	// gradual concept drift stream
	    public InstanceStream createHyperplaneStream(double magnitude, int seed, int classIndex) {
	        HyperplaneBalanced hyper = new HyperplaneBalanced();
	        hyper.magChangeOption.setValue(magnitude);
	        hyper.numClassesOption.setValue(2);
	        hyper.instanceRandomSeedOption.setValue(seed);
	        hyper.desiredClassOption.setValue(classIndex); // set minority class       
	        hyper.imbalanceClassesOption.setValue(true); // create imbalanced hyperplane
	        hyper.imbalanceWeightOption.setValue(0.1); // 10% imbalance
	        hyper.prepareForUse();

	        checkImbalance(hyper, 10000);
	        return hyper;
	    }
	 
	    private void checkImbalance(InstanceStream stream, int streamSize) {
	        int numInstances = streamSize;
	        //((OptionHandler) stream).prepareForUse(); // prepare stream   

	        int totalSamples = 0;
	        int class0 = 0;
	        int class1 = 0;

	        while (stream.hasMoreInstances() && totalSamples < 100000) {
	            //System.out.println("sample " + totalSamples);
	            totalSamples++;
	            Instance inst = stream.nextInstance();
	            if (inst.classValue() == 0) {
	                class0++;
	            } else if (inst.classValue() == 1) {
	                class1++;
	            } else {
	                System.out.println("class " + inst.classValue());
	            }

	        }

	        System.out.println("class0 " + class0 + " , class1 " + class1);

	    }
	    
	private Instances applySMOTE(Instances sample) throws Exception {

		// System.out.println("starting applying smote...");
		// System.out.println(sample.attributeStats(sample.numAttributes() -
		// 1));

		/*
		 * ignore if there is only one class
		 */
		int distinctCount = sample.attributeStats(sample.classIndex()).distinctCount;
		if (distinctCount == 1 || !this.performSmote) {
			return sample;
		}

		/*
		 * ratio   1:1 for 20:40, 100%
		 * ratio 0.9:1 for 20:40, 80%
		 */
		AttributeStats attributeStats = sample.attributeStats(sample.classIndex());
		System.out.println(attributeStats.toString());
		double ratio = (double)attributeStats.nominalCounts[0]/attributeStats.nominalCounts[1];
		double increasedPercentage = 0;
		if (ratio/desiredClassRatio > 1)
			increasedPercentage =(double) (ratio/desiredClassRatio - 1);
		else
			increasedPercentage =(double) (desiredClassRatio/ratio - 1);
			
		smote.setPercentage(increasedPercentage*100);
		System.out.println("SMOTE minority class increased percentage:"+increasedPercentage*100);
		smote.setInputFormat(sample);
		Instances newDataset = Filter.useFilter(sample, smote);
		
		System.out.println(newDataset.attributeStats(sample.classIndex()));

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

	public void setDesiredClassRatio(double desiredClassRatio) {
		this.desiredClassRatio = desiredClassRatio;
	}

	public void setPerformSMOTE(boolean performSmote) {
		this.performSmote = performSmote;
		
	}

	public InstanceStream getStream() {
		return stream;
	}

	public void setStream(InstanceStream stream) {
		this.stream = stream;
	}

	public void setTestStream(InstanceStream testStream) {
		this.testStream = testStream;
		
	}
}