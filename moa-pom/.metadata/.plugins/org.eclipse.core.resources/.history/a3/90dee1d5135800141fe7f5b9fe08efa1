package datastream.streams;

import java.util.Random;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.streams.generators.HyperplaneGenerator;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * Stream generator for Hyperplane data stream.
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */

// generates an imbalanced Hyperplane stream for binary classes
public class HyperplaneImbalanced extends HyperplaneGenerator {

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'x', "Balance the number of instances of each class."); // added

    public FlagOption imbalanceClassesOption = new FlagOption("imbalanceClasses",
            'y', "Imbalance the number of instances of each class."); // added

    public FloatOption imbalanceWeightOption = new FloatOption("weight",
            'z', "Weight of class imbalance", 0.0); // added

    protected boolean nextClassShouldBeZero;

    protected double imbalanceWeight; // imbalance percentage of class 1
    

    @Override
    public Instance nextInstance() {

        int numAtts = this.numAttsOption.getValue();
        double[] attVals = new double[numAtts + 1];
        int classLabel = 0;

        boolean desiredClassFound = false;
        while (!desiredClassFound) {
            double sum = 0.0;
            double sumWeights = 0.0;
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

            if (!this.balanceClassesOption.isSet()) {
                double rand = this.instanceRandom.nextDouble();
                if (rand < this.imbalanceWeight) {
                    if (classLabel == 1) {
                        desiredClassFound = true;
                    }
                } else {
                    if (classLabel == 0) {
                        desiredClassFound = true;
                    }
                }
            } else {
                // balance the classes      
                if ((this.nextClassShouldBeZero && (classLabel == 0)) || (!this.nextClassShouldBeZero && (classLabel == 1))) {
                    desiredClassFound = true;
                    this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                }
            }
        }
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

}
