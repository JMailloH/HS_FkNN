package org.apache.spark.mllib.classification.LHS_FkNN

import org.apache.spark.rdd._
import org.apache.spark.ml.feature.LabeledPoint
import scala.collection.mutable.ArrayBuffer

/**
 * @author Jesus Maillo
 */

class LHS_FkNN(train: RDD[LabeledPoint], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int) extends Serializable {
  //Trainning set with membership
  var trainFuzzified: RDD[Array[Double]] = null

  def getTrain: RDD[LabeledPoint] = train
  def getK: Int = k
  def getDistanceType: Int = distanceType
  def getNumClass: Int = numClass
  def getNumFeatures: Int = numFeatures
  def getNumPartitionMap: Int = numPartitionMap
  def getTrainFuzzified: RDD[Array[Double]] = trainFuzzified


  /**
    * @brief Calculate the class membership degree. Train vs train.
    */
  def LHS_Membership(): LHS_FkNN = {
    trainFuzzified = train.repartition(numPartitionMap).mapPartitions(split => knnMembership(split)).cache
    trainFuzzified.count
    this
  }

  /**
   * @brief Calculate the class membership degree vector with the information of the k nearest neighbors and return the sample with the new class membership.
   *
   * @param sample The sample with the label.
   * @param neighs The k nearest neighbors.
   */
  def fuzzify(sample: LabeledPoint, neighs: Array[Array[Float]]): Array[Double] = {
    val instance = sample.features.toArray
    val label = sample.label
    val size = instance.length
    var instMembership = new Array[Double](size + numClass)

    // Initialize the membership
    var membership = new Array[Double](numClass)
    for (i <- 0 until numClass)
      membership(i) = 0
    for (j <- 0 until k)
      membership(neighs(j)(1).toInt) = membership(neighs(j)(1).toInt) + 1

    // Calculate membership
    for (t <- 0 until numClass)
      if (label == t)
        membership(t) = 0.51 + (membership(t) / k.toDouble) * 0.49
      else
        membership(t) = (membership(t) / k.toDouble) * 0.49

    //New instance to be returned
    var w = 0
    while (w < size) {
      instMembership(w) = instance(w).toFloat
      w = w + 1
    }
    //Add the membership
    var x = 0
    while (x < numClass) {
      instMembership(w) = membership(x).toFloat
      w = w + 1
      x = x + 1
    }

    //Return instance with membership and without label
    instMembership

  }

  /**
   * @brief Calculate the class membership degree.
   *
   * @param iter Data that iterate the RDD of the train set
   */
  def knnMembership[T](iter: Iterator[LabeledPoint]): Iterator[Array[Double]] = {
    // Initialization
    var train = new ArrayBuffer[LabeledPoint]

    //Join the train set
    while (iter.hasNext)
      train.append(iter.next)
    val size = train.length

    var knnMemb = new KNN(train, k, Distance.Euclidean, numClass)

    var neigh = new Array[Array[Array[Float]]](size)
    var result: Array[Array[Double]] = new Array[Array[Double]](size)

    for (i <- 0 until size) {
      neigh(i) = knnMemb.neighbors(train(i).features)
      result(i) = fuzzify(train(i), neigh(i))
    }

    result.iterator
  }

}

object LHS_FkNN {
  /**
   * @brief Initial setting necessary.
   *
   * @param train Data that iterate the RDD of the train set
   * @param k number of neighborsThe test set in a broadcasting
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2
   * @param numClass Number of classes of the output variable
   * @param numFeatures Number of input variables
   * @param numPartitionMap Number of partition. Number of map tasks
   */
  def LHS_Membership(train: RDD[LabeledPoint], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int) = {
    new LHS_FkNN(train, k, distanceType, numClass, numFeatures, numPartitionMap).LHS_Membership()
  }
}