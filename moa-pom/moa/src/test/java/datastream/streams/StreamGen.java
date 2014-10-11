
package datastream.streams;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.tasks.WriteStreamToARFFFile;
import weka.core.Instance;

public class StreamGen {

    private static BufferedWriter writer;
    
    private static final int POSITION = 300000; // position of abrupt drift 
    private static final int WIDTH = 50000; // width of abrupt drift (use this for less abrupt sigmoidal drifts)
    private static final int ALPHA = 90; // angle of abrupt drift (use this for more abrupt drifts) 
    
    private static final double SPEED = 0.01; // speed of gradual drift
    private static final int CENTROIDS = 3; // number of centroids with drift for gruadual drift stream
     
    public static void main(String[] args) throws IOException {
        //writer = new BufferedWriter(new FileWriter("class1.txt"));
        double imbalance = 0.5;
        for (int i = 0; i < 1; i++) { 
            createImbalancedStaggerDriftStream(imbalance, POSITION, WIDTH, ALPHA, false, i);
            createRBFDriftStream(imbalance, CENTROIDS, SPEED, i);
            createImbalancedRBFStream(imbalance, i);
        }
        // writer.close();
    }
    
    public static RBFImbalanced createImbalancedRBFStream(double imbalance, int seed) {
        RBFImbalanced rbf = new RBFImbalanced();
        rbf.numClassesOption.setValue(2);
        rbf.instanceRandomSeedOption.setValue(seed);
        rbf.imbalanceWeightOption.setValue(imbalance);
        rbf.prepareForUse();

        return rbf;
    }
    
    public static RBFDrift createRBFDriftStream(double imbalance, int centroids, double speed, int seed) {
        RBFDrift rbf = new RBFDrift();
        rbf.numClassesOption.setValue(2);
        rbf.instanceRandomSeedOption.setValue(seed);
        rbf.imbalanceWeightOption.setValue(imbalance);

        rbf.numCentroidsOption.setValue(centroids); // num centriods
        rbf.numDriftCentroidsOption.setValue(centroids); // drift centriods
        rbf.speedChangeOption.setValue(speed); // speed of drift

        rbf.prepareForUse();

        return rbf;
    }

    public static InstanceStream createImbalancedStaggerNoDriftStream(double imbalance, int f, int seed) {
        StaggerImbalanced stagger = new StaggerImbalanced();
        stagger.imbalanceWeightOption.setValue(imbalance);
        stagger.imbalanceClassesOption.setValue(true);
        stagger.functionOption.setValue(f); // use function f
        stagger.instanceRandomSeedOption.setValue(seed); // stream seed
        stagger.prepareForUse();

        return stagger;
    }

    /*
    * The imbalance percentage of this output stream is higher than expected (not safe to use!)    
    */
    public static InstanceStream createHyperplaneStream(double imbalance, double magnitude, int seed) {
        HyperplaneImbalanced hyper = new HyperplaneImbalanced();
        hyper.magChangeOption.setValue(magnitude);
        hyper.numClassesOption.setValue(2);
        hyper.instanceRandomSeedOption.setValue(seed);

        hyper.imbalanceClassesOption.setValue(true); // create imbalanced hyperplane        
        hyper.imbalanceWeightOption.setValue(imbalance); // % imbalance

        hyper.prepareForUse();

        return hyper;
    }

    public static InstanceStream createImbalancedStaggerDriftStream(double imbalance, int pos, int width, double alpha, boolean useWidth, int seed) {
        InstanceStream s1 = createImbalancedStaggerNoDriftStream(imbalance, 1, seed); // function 1
        InstanceStream s2 = createImbalancedStaggerNoDriftStream(imbalance, 2, seed+1); // function 2
        return createAbruptDriftStream(s1, s2, pos, width, alpha, useWidth, seed);
    }
    
    // STAGGER stream with two suddens drift points
    // Note: this stream has not been tested
    public static InstanceStream createImbalancedStaggerTwoAbruptDrifts(double imbalance, int pos1, int pos2, int seed) {
        InstanceStream s1 = createImbalancedStaggerNoDriftStream(imbalance, 3, seed); // function 3
        InstanceStream s2 = createImbalancedStaggerDriftStream(imbalance, pos2, 0, 90.0, false, seed+1);
        
        return createAbruptDriftStream(s1, s2, pos1, 0, 90.0, false, seed+2); // join streams
    }
    
    public static InstanceStream createAbruptDriftStream(InstanceStream s1, InstanceStream s2, int position, int width, double alpha, boolean useWidth, int seed) {
        ConceptDriftStream driftStream = new ConceptDriftStream();
        
        driftStream.streamOption.setCurrentObject(s1); // combines two streams        
        driftStream.driftstreamOption.setCurrentObject(s2); // set drift stream
        driftStream.positionOption.setValue(position);

        driftStream.randomSeedOption.setValue(seed); // seed for combining streams
        
        if (useWidth) {
            driftStream.widthOption.setValue(width);
        } else {
            driftStream.alphaOption.setValue(alpha); 
        }
        
        driftStream.prepareForUse();
        return driftStream;
    }
        

    public static void checkImbalance(InstanceStream stream, int streamSize, int interval) throws IOException {
        //((OptionHandler) stream).prepareForUse(); // prepare stream        
        int totalSamples = 0;
        int class0 = 0;
        int class1 = 0;
        while (stream.hasMoreInstances() && totalSamples < streamSize) {
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
            /*
            double ratio = (double) class1 / (double) class0;
            if (interval != 0 && (totalSamples % interval == 0)) {
                System.out.println("class0 " + class0 + " , class1 " + class1 + ", ratio " + ratio);
                writer.write(class1 + "\n");
                class0 = 0;
                class1 = 0;
            }
            */
        }
        System.out.println("class0 " + class0 + " , class1 " + class1);
    }

    public static void writeStream(InstanceStream st, String filename) {
        WriteStreamToARFFFile file = new WriteStreamToARFFFile();
        file.streamOption.setCurrentObject(st);
        file.maxInstancesOption.setValue(1000000); // size of stream to write
        file.arffFileOption.setValue(filename + "_stream.arff");
        file.doTask();
    }
}
