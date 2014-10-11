package datastream.streams;

import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import moa.core.MiscUtils;
import moa.options.FloatOption;
import moa.streams.generators.*;

// modified to generate imbalanced RBF stream 
public class RBFImbalanced extends RandomRBFGenerator {

    public FloatOption imbalanceWeightOption = new FloatOption("weight",
            'y', "Weight of class imbalance", 0.0);
    
    protected double weight;
    
    
    @Override
    public Instance nextInstance() {
        int useThis = 0;
        Instance inst = new DenseInstance(0);
        Random rand = new Random();

        while (useThis == 0) {
            Centroid centroid = this.centroids[MiscUtils.chooseRandomIndexBasedOnWeights(this.centroidWeights,
                    this.instanceRandom)];
            int numAtts = this.numAttsOption.getValue();
            double[] attVals = new double[numAtts + 1];
            for (int i = 0; i < numAtts; i++) {
                attVals[i] = (this.instanceRandom.nextDouble() * 2.0) - 1.0;
            }
            double magnitude = 0.0;
            for (int i = 0; i < numAtts; i++) {
                magnitude += attVals[i] * attVals[i];
            }
            magnitude = Math.sqrt(magnitude);
            double desiredMag = this.instanceRandom.nextGaussian()
                    * centroid.stdDev;
            double scale = desiredMag / magnitude;
            for (int i = 0; i < numAtts; i++) {
                attVals[i] = centroid.centre[i] + attVals[i] * scale;
            }

            inst = new DenseInstance(1.0, attVals);
            inst.setDataset(getHeader());
            inst.setClassValue(centroid.classLabel);

            if (inst.classValue() != 1.0 || (rand.nextDouble() < (weight / (1 - weight))) ) {
                useThis = 1;
            }
        }

        return inst;
    }

    
    @Override
    public void restart() {
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        this.weight = this.imbalanceWeightOption.getValue();
    }
}
