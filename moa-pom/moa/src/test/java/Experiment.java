
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

    private static final int CLASS_INDEX = 1; // index of minority class
    private double desiredClassRatio;
    private static Instances data;
    private static SMOTE smote;

    private Classifier learner;
    private int numberSamplesCorrect;
    private int sampleSize = 2000;
    private boolean performSmote = true;
    
    private static String currentArffAbsolutePath;
    private String filename;
    private InstanceStream stream;
    private InstanceStream testStream;
    private BufferedWriter bw;

    private long startTime; // initial time
    private long elapsedTime; // how much time has passed till most recent stop time
    private long lastStartTime; // current runtime values ignores the time used in testing

    public Experiment() {
    }

    public void run(int numInstances, boolean isTesting, String csvFileName) throws Exception {
        filename = csvFileName;
        bw = new BufferedWriter(new FileWriter(csvFileName + ".csv"));
        bw.write("sampleStartIndex\tmemory\truntime\taccuracy\tprecision\trecall\tfScore\n"); // write header to file
        // stream = new ArffFileStream(currentArffAbsolutePath, -1);
        
        ((OptionHandler) this.stream).prepareForUse();
        ((OptionHandler) this.testStream).prepareForUse();
        
        learner.setModelContext(this.stream.getHeader());
        learner.prepareForUse();
        
        int numberSamples = 0;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        startTime = TimingUtils.getNanoCPUTimeOfCurrentThread(); // initial start time
        elapsedTime = (long) 0.0; // no time passed
        lastStartTime = startTime;
        
        /*
         * Generate a fixed size sample
         */
        int curentSize = 0;
        int startBucketIndex = 0; // index of start of bucket on pre-SMOTE stream
        Instances sample = new Instances(this.stream.getHeader(), sampleSize);
        
        while (this.stream.hasMoreInstances() && numberSamples < numInstances) {
            if (curentSize < sampleSize) {
                Instance trainInst = this.stream.nextInstance();
                numberSamples++;
                sample.add(trainInst);
                curentSize++;
            } else {
                Instances newDataset = applySMOTE(sample);
                long currentTime = TimingUtils.getNanoCPUTimeOfCurrentThread(); // time before testing
                elapsedTime = elapsedTime + currentTime - lastStartTime; // stop timer when testing
                if (isTesting) {
                    testPerformance(sample, sampleSize, startBucketIndex); // test before training
                }
                currentTime = TimingUtils.getNanoCPUTimeOfCurrentThread(); // time after testing 
                lastStartTime = currentTime; // continue timer after testing
                training(newDataset);
                startBucketIndex = numberSamples;
                curentSize = 0;
                sample.clear();
            }
        }
        
        /*
         * Last sample if it still has some instances in it
         */
        if (!sample.isEmpty()) {
            Instances newDataset = applySMOTE(sample);
            long currentTime = TimingUtils.getNanoCPUTimeOfCurrentThread(); // time before testing
            elapsedTime = elapsedTime + currentTime - lastStartTime; // stop timer when testing
            if (isTesting) {
                testPerformance(sample, sampleSize, startBucketIndex); // test before training
            }
            currentTime = TimingUtils.getNanoCPUTimeOfCurrentThread(); // time after testing
            lastStartTime = currentTime; // continue timer after adding
            training(newDataset);
            curentSize = 0;
            sample.clear();
        }

        long currentTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        elapsedTime = elapsedTime + currentTime - lastStartTime;
        double totalTime = TimingUtils.nanoTimeToSeconds(elapsedTime); // total time
        System.out.println(csvFileName + " total time: " + totalTime);

        int byteSize = learner.measureByteSize(); // total memory used by learner
        System.out.println("end memory: " + byteSize);

        bw.close(); // close output file
    }

    private double calculatePrecision(int tP, int fP) {
        if (tP + fP == 0.0) {
            return 0.0;
        }
        return (double) tP / (double) (tP + fP);
    }

    private double calculateRecall(int tP, int fN) {
        if (tP + fN == 0) {
            return 0.0;
        }
        return (double) tP / (double) (tP + fN);
    }

    private double calculateF1(double precision, double recall) {
        if (precision + recall == 0.0) {
            return 0.0;
        }
        return (double) (2 * precision * recall) / (double) (precision + recall);
    }

    private double calculateAccuracy(int numCorrect, int totalNum) {
        return 100.0 * (double) (numCorrect) / (double) (totalNum);
    }

    private void testPerformance(Instances sample, int numberSamples, int startIndex) throws IOException {
        numberSamplesCorrect = 0;
        int tP = 0, tN = 0, fP = 0, fN = 0;
        int index = 0;

        for (Instance instance : sample) {
            index++;
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
        int byteSize = learner.measureByteSize();

        // time used in training and SMOTE (excludes testing and stream generation time)
        double runtime = TimingUtils.nanoTimeToSeconds(elapsedTime);

        //System.err.println(startIndex + " " + filename + " mem: " + byteSize);
        //System.err.println(startIndex + " " + filename + " elapsed time: " + runtime);        
        //System.err.println(filename + " " + startIndex + " tP " + tP + ", tN " + tN + ", fP " + fP + ", fN " + fN);
        bw.write(startIndex + "\t" + byteSize + "\t" + runtime + "\t" + accuracy + "\t" + precision + "\t" + recall + "\t" + f1 + "\n");

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
         * ratio 1:1 for 20:40, 100%
         * ratio 0.9:1 for 20:40, 80%
         */
        //System.out.println("Stream: " + filename);
        AttributeStats attributeStats = sample.attributeStats(sample.classIndex());
        //System.out.println(attributeStats.toString());
        double ratio = (double) attributeStats.nominalCounts[0] / attributeStats.nominalCounts[1];
        double increasedPercentage = 0;
        if (ratio / desiredClassRatio > 1) {
            increasedPercentage = (double) (ratio / desiredClassRatio - 1);
        } else {
            increasedPercentage = (double) (desiredClassRatio / ratio - 1);
        }
        if (increasedPercentage * 100 < 5) {
            return sample;
        }
        smote.setPercentage(increasedPercentage * 100);
        //System.out.println("SMOTE minority class increased percentage:" + increasedPercentage * 100);
        smote.setInputFormat(sample);
        Instances newDataset = Filter.useFilter(sample, smote);
        //System.out.println(newDataset.attributeStats(sample.classIndex()));
        return newDataset;
    }

    private void training(Instances newDataset) {
        for (Instance trainInst : newDataset) {
            learner.trainOnInstance(trainInst);
        }
    }

    public static void setSmote(SMOTE smote) {
        Experiment.smote = smote;
    }

    public void setCurrentArffAbsolutePath(String currentArffAbsolutePath2) {
        Experiment.currentArffAbsolutePath = currentArffAbsolutePath2;
    }

    public void setData(Instances dataSet) {
        Experiment.data = dataSet;
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
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
