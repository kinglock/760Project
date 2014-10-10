
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

/**
 * Stream generator for Hyperplane data stream.
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
// generates a balanced Hyperplane stream
public class HyperplaneBalanced extends AbstractOptionHandler implements
        InstanceStream {

    @Override
    public String getPurposeString() {
        return "Generates a problem of predicting class of a rotating hyperplane.";
    }

    private static final long serialVersionUID = 1L;

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public IntOption numClassesOption = new IntOption("numClasses", 'c',
            "The number of classes to generate.", 2, 2, Integer.MAX_VALUE);

    public IntOption numAttsOption = new IntOption("numAtts", 'a',
            "The number of attributes to generate.", 10, 0, Integer.MAX_VALUE);

    public IntOption numDriftAttsOption = new IntOption("numDriftAtts", 'k',
            "The number of attributes with drift.", 2, 0, Integer.MAX_VALUE);

    public FloatOption magChangeOption = new FloatOption("magChange", 't',
            "Magnitude of the change for every example", 0.0, 0.0, 1.0);

    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 5, 0, 100);

    public IntOption sigmaPercentageOption = new IntOption("sigmaPercentage",
            's', "Percentage of probability that the direction of change is reversed.", 10, 0, 100);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'x', "Balance the number of instances of each class."); // added    

    public FlagOption imbalanceClassesOption = new FlagOption("imbalanceClasses",
            'y', "Imbalance the number of instances of each class."); // added

    public FloatOption imbalanceWeightOption = new FloatOption("weight",
            'z', "Weight of class imbalance", 0.0); //added

    public IntOption desiredClassOption = new IntOption(
            "desiredClassIndex", 'd',
            "Index of the class of interest", 1); // added
    
    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    protected double[] weights;

    protected int[] sigma;

    public int numberInstance;

    protected boolean nextClassShouldBeZero;

    protected double imbalanceWeight;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        monitor.setCurrentActivity("Preparing hyperplane...", -1.0);
        generateHeader();
        restart();
    }

    protected void generateHeader() {
        FastVector attributes = new FastVector();
        for (int i = 0; i < this.numAttsOption.getValue(); i++) {
            attributes.addElement(new Attribute("att" + (i + 1)));
        }

        FastVector classLabels = new FastVector();
        for (int i = 0; i < this.numClassesOption.getValue(); i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
    }

    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }

    @Override
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }

    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public Instance nextInstance() {

        int numAtts = this.numAttsOption.getValue();
        double[] attVals = new double[numAtts + 1];
        double sum = 0.0;
        double sumWeights = 0.0;

        boolean desiredClassFound = false;
        int classLabel = -1; // default value

        while ((imbalanceClassesOption.isSet() && !desiredClassFound) || (balanceClassesOption.isSet() && !desiredClassFound)) {
            for (int i = 0; i < numAtts; i++) {
                attVals[i] = this.instanceRandom.nextDouble();
                sum += this.weights[i] * attVals[i];
                sumWeights += this.weights[i];
            }

            if (sum >= sumWeights * 0.5) {
                classLabel = 1;
            } else {
                classLabel = 0;
            }

            // balance the classes
            if ((this.nextClassShouldBeZero && (classLabel == 0)) || (!this.nextClassShouldBeZero && (classLabel == 1))) {
                desiredClassFound = true;
                this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                // then imbalance classes if required
                if (imbalanceClassesOption.isSet()) {
                    if (classLabel != desiredClassOption.getValue() || this.instanceRandom.nextDouble() < (imbalanceWeight / (1 - imbalanceWeight))) {
                        desiredClassFound = true;
                    } else {
                        desiredClassFound = false; // keep searching
                    }
                }
            }
        } // else keep searching

        //Add Noise
        if ((1 + (this.instanceRandom.nextInt(100))) <= this.noisePercentageOption.getValue()) {
            classLabel = (classLabel == 0 ? 1 : 0);
        }

        Instance inst = new DenseInstance(1.0, attVals);

        inst.setDataset(getHeader());
        inst.setClassValue(classLabel);

        addDrift();
        return inst;
    }

    private void addDrift() {
        for (int i = 0; i < this.numDriftAttsOption.getValue(); i++) {
            this.weights[i] += (double) ((double) sigma[i]) * ((double) this.magChangeOption.getValue());
            if (//this.weights[i] >= 1.0 || this.weights[i] <= 0.0 ||
                    (1 + (this.instanceRandom.nextInt(100))) <= this.sigmaPercentageOption.getValue()) {
                this.sigma[i] *= -1;
            }
        }
    }

    @Override
    public void restart() {
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        this.weights = new double[this.numAttsOption.getValue()];
        this.sigma = new int[this.numAttsOption.getValue()];
        for (int i = 0; i < this.numAttsOption.getValue(); i++) {
            this.weights[i] = this.instanceRandom.nextDouble();
            this.sigma[i] = (i < this.numDriftAttsOption.getValue() ? 1 : 0);
        }
        this.nextClassShouldBeZero = false;
        this.imbalanceWeight = imbalanceWeightOption.getValue();
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }
}
