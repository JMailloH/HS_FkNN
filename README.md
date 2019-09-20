# HS_FkNN: Hybrid Spill Tree Fuzzy k Nearest Neighbors.

This is an open-source Spark package about two Fuzzy k Nearest Neighbors classifier based on Apache Spark. We take advantage of its in-memory operations to simultaneously classify big amounts of unseen cases against a big training dataset. It consists of two stages: class membership degree and classification. The class membership degree stage changes the class label to class membership degree vector. Two approaches are proposed to address this stage: Local Hybrid Spill Tree FkNN (LHS_FkNN) and Global Approximate Hybrid Spill Tree FkNN (GAHS-FkNN). LHS_FkNN follows a local approach, which calculates membership independently on each partition. GAHS-FkNN has a global approach based on Hybrid Spill Tree, considering all data from the training set. The classification stage is common to both models and is based on Hybrid Spill Tree.

## Cite this software as:

J. Maillo, S. García, J. Luengo, F. Herrera and I. Triguero, "Fast and Scalable Approaches to Accelerate the Fuzzy k Nearest Neighbors Classifier for Big Data" in IEEE Transactions on Fuzzy Systems. doi: 10.1109/TFUZZ.2019.2936356

# How to use

## Pre-requiriments and software version
The following software have to get installed:
- Scala. Version 2.11
- Spark. Version 2.3.2
- Maven. Version 3.5.2
- JVM. Java Virtual Machine. Version 1.8.0 because Scala run over it.

## Download and build with maven
- Download source code: It is host on GitHub. To get the sources and compile them we will need the next git instruction.
```git clone https://github.com/JMailloH/HS_FkNN.git ```
- Build jar file: Once we have the sources, we generate the .jar file of the project by the next maven instruction.
```mvn package -Dmaven.test.skip=true. ```

Another alternative is download by spark-package at [https://spark-packages.org/package/JMailloH/HS_FkNN](https://spark-packages.org/package/JMailloH/HS_FkNN)


## How to run

The execution of this software can be considered as 3 independent parts: local membership (LHS-Memb), global membership (GAHS-Memb) and classification. For more details see the example execution in the file [runHS_FkNN.scala](https://github.com/JMailloH/HS_FkNN/tree/master/src/main/scala/run/runHS_FkNN.scala).

The run.sh file contains an example call for each algorithm. A generic sample of run could be: 

/opt/spark-2.3.2/bin/spark-submit --master "URL" --class org.apache.spark.run.runHS_FkNN ./target/HS_FkNN-1.0.jar "path-to-header" "path-to-train" "path-to-test" "number-of-neighbors" "number-of-maps" "stage" "path-to-output"

- ```--class org.apache.spark.run.runHS_FkNN ./target/HS_FkNN-1.0.jar``` Determine the jar file to be run.
- ```"path-to-header"``` Path from HDFS to header.
- ```"path-to-train"``` Path from HDFS to training set.
- ```"path-to-test"``` Path from HDFS to test set.
- ```"number-of-neighbors"```Number of neighbors. The value of k.
- ```"number-of-maps"``` Number of map tasks.
- ```"stage"``` Select the stage to be executed: GAHS-FkNN or LHS-FkNN or classification
- ```"path-to-output"``` Path from HDFS to the output.

## Output

The output is different in the membership or classification stage. The membership stage writes the Fuzzy Training Set to HDFS in the same directory as the Training Set. It also writes the runtime in the directory specified at "path-to-output" parameter. The classification stage returns the runtime, accuracy, and AUC in the directory specified at "path-to-output" parameter.
