package datastream.streams;

import moa.core.InstancesHeader;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.generators.STAGGERGenerator;
import weka.core.DenseInstance;
import weka.core.Instance;

// modified to generate imbalanced STAGGER stream 
public class StaggerImbalanced extends STAGGERGenerator {

    public FlagOption imbalanceClassesOption = new FlagOption("imbalanceClasses",
            'x', "Imbalance the number of instances of each class.");

    public FloatOption imbalanceWeightOption = new FloatOption("weight",
            'y', "Weight of class imbalance", 0.0);

    public IntOption desiredClassOption = new IntOption(
            "desiredClassIndex", 'z',
            "The class of interest (the minority class)", 1);

    protected double imbalanceWeight; // class 1's imbalance percentage

    protected int desiredClass; // defaults to 1

    @Override
    public Instance nextInstance() {

        int size = 0, color = 0, shape = 0, group = 0;
        boolean desiredClassFound = false;
        while (!desiredClassFound) {
            // generate attributes
            size = this.instanceRandom.nextInt(3);
            color = this.instanceRandom.nextInt(3);
            shape = this.instanceRandom.nextInt(3);

            // determine class
            group = classificationFunctions[this.functionOption.getValue() - 1].determineClass(size, color, shape);
            if (!this.balanceClassesOption.isSet() && !this.imbalanceClassesOption.isSet()) { // modified
                desiredClassFound = true;
            } else {
                // balance the classes
                if ((this.nextClassShouldBeZero && (group == 0)) || (!this.nextClassShouldBeZero && (group == 1))) {
                    desiredClassFound = true;
                    this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                    if (this.imbalanceClassesOption.isSet()) { // then imbalance the classes
                        if (group != this.desiredClassOption.getValue() || this.instanceRandom.nextDouble() < (this.imbalanceWeight / (1 - this.imbalanceWeight))) {
                            desiredClassFound = true;
                        } else {
                            desiredClassFound = false; // keep searching                           
                        }
                    }
                } // else keep searching
            }
        }

        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setValue(0, size);
        inst.setValue(1, color);
        inst.setValue(2, shape);
        inst.setDataset(header);
        inst.setClassValue(group);
        return inst;
    }

    @Override
    public void restart() {
        super.restart();
        this.imbalanceWeight = imbalanceWeightOption.getValue();
        this.desiredClass = desiredClassOption.getValue();
    }
}
