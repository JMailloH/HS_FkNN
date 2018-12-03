package org.apache.spark.ml.classification

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.HS_FkNN._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.{ Identifiable, SchemaUtils }
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ DoubleType, StructType }
import org.apache.spark.sql.{ DataFrame, Dataset, Row }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkException
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable

class HS_FkNNClassifier(override val uid: String) extends ProbabilisticClassifier[Vector, HS_FkNNClassifier, HS_FkNNClassificationModel]
    with HS_FkNNParams with HasWeightCol {

  def this() = this(Identifiable.randomUID("knnc"))

  /** @group setParam */
  override def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  override def setLabelCol(value: String): this.type = {
    set(labelCol, value)

    if ($(weightCol).isEmpty) {
      set(inputCols, Array(value))
    } else {
      set(inputCols, Array(value, $(weightCol)))
    }
  }

  //fill in default label col
  setDefault(inputCols, Array($(labelCol)))

  /** @group setWeight */
  def setWeightCol(value: String): this.type = {
    set(weightCol, value)

    if (value.isEmpty) {
      set(inputCols, Array($(labelCol)))
    } else {
      set(inputCols, Array($(labelCol), value))
    }
  }

  setDefault(weightCol -> "")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)
  //final def getK: Int = $(k)

  /** @group setParam */
  def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

  /** @group setParam */
  def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

  /** @group setParam */
  def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

  /** @group setParam */
  def setBufferSizeSampleSizes(value: Array[Int]): this.type = set(bufferSizeSampleSizes, value)

  /** @group setParam */
  def setBalanceThreshold(value: Double): this.type = set(balanceThreshold, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  protected def train(dataset: Dataset[_], numClasses: Int): HS_FkNNClassificationModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val knnModel = copyValues(new HS_FkNN()).fit(dataset)
    knnModel.toNewClassificationModel(uid, numClasses)
  }

  override protected def train(dataset: Dataset[_]): HS_FkNNClassificationModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSummarizer = instances.treeAggregate(
      new MultiClassSummarizer)(
        seqOp = (c, v) => (c, v) match {
          case (labelSummarizer: MultiClassSummarizer, (label: Double, features: Vector)) =>
            labelSummarizer.add(label)
        },
        combOp = (c1, c2) => (c1, c2) match {
          case (classSummarizer1: MultiClassSummarizer, classSummarizer2: MultiClassSummarizer) =>
            classSummarizer1.merge(classSummarizer2)
        })

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length

    if (numInvalid != 0) {
      val msg = s"Classification labels should be in {0 to ${numClasses - 1} " +
        s"Found $numInvalid invalid labels."
      logError(msg)
      throw new SparkException(msg)
    }

    val knnModel = copyValues(new HS_FkNN()).fit(dataset)
    knnModel.toNewClassificationModel(uid, numClasses)
  }

  def trainFuzzy(dataset: Dataset[_], numClasses: Int): HS_FkNNClassificationModel = {

    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val knnModel = copyValues(new HS_FkNN()).fitFuzzy(dataset.select($(featuresCol), $(rawPredictionCol)))
    knnModel.toNewClassificationModel(uid, numClasses)

  }

  def fit(dataset: Dataset[_], numClasses: Int): HS_FkNNClassificationModel = {
    // Need to overwrite this method because we need to manually overwrite the buffer size
    // because it is not supposed to stay the same as the Classifier if user sets it to -1.
    val model = train(dataset, numClasses)
    val bufferSize = model.getBufferSize
    copyValues(model.setParent(this)).setBufferSize(bufferSize)
  }

  def fitFuzzy(dataset: Dataset[_], numClasses: Int): HS_FkNNClassificationModel = {
    // Need to overwrite this method because we need to manually overwrite the buffer size
    // because it is not supposed to stay the same as the Classifier if user sets it to -1.
    val model = trainFuzzy(dataset, numClasses)
    val bufferSize = model.getBufferSize
    copyValues(model.setParent(this)).setBufferSize(bufferSize)
  }

  override def copy(extra: ParamMap): HS_FkNNClassifier = defaultCopy(extra)
}

class HS_FkNNClassificationModel private[ml] (
  override val uid: String,
  val topTree: Broadcast[Tree],
  val subTrees: RDD[Tree],
  val _numClasses: Int) extends ProbabilisticClassificationModel[Vector, HS_FkNNClassificationModel]
    with HS_FkNNModelParams with HasWeightCol with Serializable {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  override def numClasses: Int = _numClasses

  def transformPredict(dataset: Dataset[_]): RDD[(Double, Double)] = {
    dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
      .leftOuterJoin(transform(dataset, topTree, subTrees).map {
        case (id, membershipDistance) =>
          val values = new ArrayBuffer[Any]
          val prediction = membership2Prediction(membershipDistance)
          (id, prediction)
      })
      .map {
        case (i, (row, values)) =>
          val label = row.getDouble(0)
          val prediction = values.get
          (label, prediction)
      }
  }

  def GAHS_Membership(dataset: Dataset[_]): DataFrame = {
    val getWeight: Row => Double = {
      if ($(weightCol).isEmpty) {
        r => 1.0
      } else {
        r => r.getDouble(1)
      }
    }

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(
          transform(dataset, topTree, subTrees).map {
            case (id, labelsDists) =>
              val (labels, _) = labelsDists.unzip
              val vector = new Array[Double](numClasses)
              var i = 0
              while (i < labels.length) {
                vector(labels(i).getDouble(0).toInt) += getWeight(labels(i))
                i += 1
              }
              val rawPrediction = Vectors.dense(vector)

              val values = new ArrayBuffer[Any]
              if ($(rawPredictionCol).nonEmpty) {
                values.append(rawPrediction)
              }

              (id, values)
          })
        .map {
          case (i, (row, values)) =>
            val label = row.getDouble(0)
            val rawPredictionAny = values.get.toArray
            val rawPrediction = rawPredictionAny(0).toString().drop(1).dropRight(1).split(",").map { x => x.toDouble }

            val membership: ArrayBuffer[Any] = raw2membership(rawPrediction, label)
            Row.fromSeq(row.toSeq ++ membership)
        },
      transformSchema(dataset.schema))

  }

  override def transformSchema(schema: StructType): StructType = {
    var transformed = schema
    if ($(rawPredictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(rawPredictionCol), new VectorUDT)
    }
    transformed
  }

  override def copy(extra: ParamMap): HS_FkNNClassificationModel = {
    val copied = new HS_FkNNClassificationModel(uid, topTree, subTrees, numClasses)
    copyValues(copied, extra).setParent(parent)
  }

  def raw2membership(rawPrediction: Array[Double], label: Double): ArrayBuffer[Any] = {
    var membershipComput = new Array[Double](numClasses)
    // Calculate membership
    for (t <- 0 until numClasses)
      if (label == t)
        membershipComput(t) = 0.51 + (rawPrediction(t) / getK.toDouble) * 0.49
      else
        membershipComput(t) = (rawPrediction(t) / getK.toDouble) * 0.49

    val membership = new ArrayBuffer[Any]
    membership.append(Vectors.dense(membershipComput))
    membership
  }

  def membership2Prediction[T](membershipDistance: Array[(Row, Double)]): Double = {
    val numClass = membershipDistance(0)._1.getAs[Vector](0).size
    var predictComput: Array[Double] = new Array[Double](numClass)

    for (i <- 0 until numClass) {
      predictComput(i) = 0
      var dist_acu = 0.0 //1 / dist  
      var dist = 0.0
      for (j <- 0 until getK) {
        dist = 1.0 / (membershipDistance(j)._2 * membershipDistance(j)._2)
        dist_acu = dist_acu + dist
        predictComput(i) = predictComput(i) + (membershipDistance(j)._1.getAs[Vector](0)(i) * dist) // i - 1 because sample(0) is the distance
      }
      predictComput(i) = predictComput(i) / dist_acu

    }

    var max: Double = 0f
    var max_pos: Int = 0
    for (i <- 0 until numClass) {
      if (predictComput(i) > max) {
        max = predictComput(i)
        max_pos = i
      }
    }

    max_pos.toDouble
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size

        var sum = 0.0
        while (i < size) {
          sum += dv.values(i)
          i += 1
        }

        i = 0
        while (i < size) {
          dv.values(i) /= sum
          i += 1
        }

        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in KNNClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }
}

/**
 * MultiClassSummarizer computes the number of distinct labels and corresponding counts,
 * and validates the data to see if the labels used for k class multi-label classification
 * are in the range of {0, 1, ..., k - 1} in an online fashion.
 *
 * Two MultilabelSummarizer can be merged together to have a statistical summary of the
 * corresponding joint dataset.
 */
private[classification] class MultiClassSummarizerMod extends Serializable {
  // The first element of value in distinctMap is the actually number of instances,
  // and the second element of value is sum of the weights.
  private val distinctMap = new mutable.HashMap[Int, (Long, Double)]
  private var totalInvalidCnt: Long = 0L

  /**
   * Add a new label into this MultilabelSummarizer, and update the distinct map.
   *
   * @param label The label for this data point.
   * @param weight The weight of this instances.
   * @return This MultilabelSummarizer
   */
  def add(label: Vector, weight: Double = 1.0): this.type = {
    require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    this
  }

  /**
   * Merge another MultilabelSummarizer, and update the distinct map.
   * (Note that it will merge the smaller distinct map into the larger one using in-place
   * merging, so either `this` or `other` object will be modified and returned.)
   *
   * @param other The other MultilabelSummarizer to be merged.
   * @return Merged MultilabelSummarizer object.
   */
  def merge(other: MultiClassSummarizerMod): MultiClassSummarizerMod = {
    val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    smallMap.distinctMap.foreach {
      case (key, value) =>
        val (counts: Long, weightSum: Double) = largeMap.distinctMap.getOrElse(key, (0L, 0.0))
        largeMap.distinctMap.put(key, (counts + value._1, weightSum + value._2))
    }
    largeMap.totalInvalidCnt += smallMap.totalInvalidCnt
    largeMap
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  /** @return The number of distinct labels in the input dataset. */
  def numClasses: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

  /** @return The weightSum of each label in the input dataset. */
  def histogram: Array[Double] = {
    val result = Array.ofDim[Double](numClasses)
    var i = 0
    val len = result.length
    while (i < len) {
      result(i) = distinctMap.getOrElse(i, (0L, 0.0))._2
      i += 1
    }
    result
  }
}

/**
 * MultiClassSummarizer computes the number of distinct labels and corresponding counts,
 * and validates the data to see if the labels used for k class multi-label classification
 * are in the range of {0, 1, ..., k - 1} in an online fashion.
 *
 * Two MultilabelSummarizer can be merged together to have a statistical summary of the
 * corresponding joint dataset.
 */
private[classification] class MultiClassSummarizer extends Serializable {
  // The first element of value in distinctMap is the actually number of instances,
  // and the second element of value is sum of the weights.
  private val distinctMap = new mutable.HashMap[Int, (Long, Double)]
  private var totalInvalidCnt: Long = 0L

  /**
   * Add a new label into this MultilabelSummarizer, and update the distinct map.
   *
   * @param label The label for this data point.
   * @param weight The weight of this instances.
   * @return This MultilabelSummarizer
   */
  def add(label: Double, weight: Double = 1.0): this.type = {
    require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    if (label - label.toInt != 0.0 || label < 0) {
      totalInvalidCnt += 1
      this
    } else {
      val (counts: Long, weightSum: Double) = distinctMap.getOrElse(label.toInt, (0L, 0.0))
      distinctMap.put(label.toInt, (counts + 1L, weightSum + weight))
      this
    }
  }

  /**
   * Merge another MultilabelSummarizer, and update the distinct map.
   * (Note that it will merge the smaller distinct map into the larger one using in-place
   * merging, so either `this` or `other` object will be modified and returned.)
   *
   * @param other The other MultilabelSummarizer to be merged.
   * @return Merged MultilabelSummarizer object.
   */
  def merge(other: MultiClassSummarizer): MultiClassSummarizer = {
    val (largeMap, smallMap) = if (this.distinctMap.size > other.distinctMap.size) {
      (this, other)
    } else {
      (other, this)
    }
    smallMap.distinctMap.foreach {
      case (key, value) =>
        val (counts: Long, weightSum: Double) = largeMap.distinctMap.getOrElse(key, (0L, 0.0))
        largeMap.distinctMap.put(key, (counts + value._1, weightSum + value._2))
    }
    largeMap.totalInvalidCnt += smallMap.totalInvalidCnt
    largeMap
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  /** @return The number of distinct labels in the input dataset. */
  def numClasses: Int = if (distinctMap.isEmpty) 0 else distinctMap.keySet.max + 1

  /** @return The weightSum of each label in the input dataset. */
  def histogram: Array[Double] = {
    val result = Array.ofDim[Double](numClasses)
    var i = 0
    val len = result.length
    while (i < len) {
      result(i) = distinctMap.getOrElse(i, (0L, 0.0))._2
      i += 1
    }
    result
  }
}
