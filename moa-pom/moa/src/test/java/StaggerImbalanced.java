
import java.util.Random;
import moa.core.InstancesHeader;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.generators.STAGGERGenerator;
import weka.core.DenseInstance;
import weka.core.Instance;

public class StaggerImbalanced extends STAGGERGenerator {

    public FlagOption imbalanceClassesOption = new FlagOption("imbalanceClasses",
            'x', "Imbalance the number of instances of each class.");

    public FloatOption imbalanceWeightOption = new FloatOption("weight",
            'y', "Weight of class imbalance", 0.0);

    public IntOption desiredClassOption = new IntOption(
            "desiredClassIndex", 'z',
            "Index of the class of interest", 1);

    protected boolean balance = true;

    protected double imbalanceWeight;

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
            if (!balance) { // modified
                desiredClassFound = true;
            } else {
                // balance the classes
                if ((this.nextClassShouldBeZero && (group == 0)) || (!this.nextClassShouldBeZero && (group == 1))) {
                    desiredClassFound = true;
                    this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                    if (imbalanceClassesOption.isSet()) {
                        if (group != desiredClassOption.getValue() || this.instanceRandom.nextDouble() < (imbalanceWeight / (1 - imbalanceWeight))) {
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
        imbalanceWeight = imbalanceWeightOption.getValue();
    }
}
