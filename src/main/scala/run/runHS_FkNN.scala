package org.apache.spark.run

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.classification.HS_FkNNClassifier
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.LHS_FkNN.LHS_FkNN
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import org.apache.log4j.Logger
import utils.keel.KeelParser

object runHS_FkNN extends Serializable {

  var sc: SparkContext = null

  def main(arg: Array[String]) {

    var logger = Logger.getLogger(this.getClass())

    if (arg.length < 6) {
      logger.error("=> wrong parameters number")
      System.err.println("Parameters \n\t<path-to-header>\n\t<path-to-train>\n\t<path-to-test>\n\t<number-of-neighbors>\n\t<number-of-partition>\n\t<satege = ''GAHS-Memb'' or ''LHS-Memb'' or ''classification''>\n\t<path-to-output>")
      System.exit(1)
    }

    //Reading parameters
    val pathHeader = arg(0)
    val pathTrain = arg(1)
    val pathTest = arg(2)
    val K = arg(3).toInt
    val numPartition = arg(4).toInt
    val stage = arg(5)
    val pathOutput = arg(6)

    //Clean pathOutput for set the jobName
    var outDisplay: String = pathOutput

    //Basic setup
    val nameDataset = pathHeader.substring(1 + pathHeader.lastIndexOf("/"), pathHeader.length - 7)
    val jobName = "FuzzykNN - Dataset: " + nameDataset + "|K: " + K + "|#Partition: " + numPartition

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    sc = new SparkContext(conf)
    // Enabled sqlContext to use .toDF()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathToHeader \"" + pathHeader + "\"")
    logger.info("=> pathToTrain \"" + pathTrain + "\"")
    logger.info("=> pathToTest \"" + pathTest + "\"")
    logger.info("=> NumberNeighbors \"" + K + "\"")
    logger.info("=> NumberPartition \"" + numPartition + "\"")
    logger.info("=> pathToOuput \"" + pathOutput + "\"")

    //Reading header of the dataset and dataset
    val converter = new KeelParser(sc, pathHeader)

    val numClasses = converter.getNumClassFromHeader()
    val numFeatures = converter.getNumFeaturesFromHeader()

    var timeBeg: Long = 0l
    var timeEnd: Long = 0l
    var timeMemberBeg: Long = 0l
    var timeMemberEnd: Long = 0l
    var timePredictBeg: Long = 0l
    var timePredictEnd: Long = 0l
    timeBeg = System.nanoTime

    if (stage == "GAHS-Memb") {
      //First stage. Compute the Global approach of the class membership degree stage
      timeMemberBeg = System.nanoTime

      val train = sc.textFile(pathTrain: String, numPartition).map(line => converter.parserToLabeledPoint(line)).toDF().persist
      val knn = new HS_FkNNClassifier().setTopTreeSize(train.count().toInt / 500).setK(K)
      val knnModel = knn.fit(train, numClasses)
      val trainWithMembership = knnModel.GAHS_Membership(train).persist
      trainWithMembership.count()

      timeMemberEnd = System.nanoTime

      // Write the output
      var writerReport = new ListBuffer[String]
      writerReport += "***Report.txt ==> Contain: Confusion Matrix; Precision; Total Runtime***\n"
      writerReport += "@MembershipRuntime\n" + (timeMemberEnd - timeMemberBeg) / 1e9
      val Report = sc.parallelize(writerReport, 1)
      Report.saveAsTextFile(pathOutput + "/Report.txt")
      trainWithMembership.rdd.map(line => line.toString).coalesce(1, true)saveAsTextFile(pathTrain.replace("tra.data", "tra-GAHS-Memb-map" + numPartition + "k" + K + ".data"))

    } else if (stage == "LHS-Memb"){
      //First stage. Compute the Local approach of the class membership degree stage
      timeMemberBeg = System.nanoTime

      val trainRaw = sc.textFile(pathTrain: String, numPartition)
      val train = trainRaw.map(line => converter.parserToLabeledPoint(line)).persist
      val knn = LHS_FkNN.LHS_Membership(train, K, 2, numClasses, numFeatures, numPartition)
      val trainWithMembership = knn.getTrainFuzzified
      trainWithMembership.count()

      timeMemberEnd = System.nanoTime

      // Write the output
      var writerReport = new ListBuffer[String]
      writerReport += "***Report.txt ==> Contain: Confusion Matrix; Precision; Total Runtime***\n"
      writerReport += "@MembershipRuntime\n" + (timeMemberEnd - timeMemberBeg) / 1e9
      val Report = sc.parallelize(writerReport, 1)
      Report.saveAsTextFile(pathOutput + "/Report.txt")
      trainWithMembership.map { line =>
        var rawFeatures: Array[Double] = line.dropRight(numClasses)
        var features = "["
        rawFeatures.foreach { x => features += x + "," }
        features = features.dropRight(1) + "],"
        var rawMembership: Array[Double] = line.drop(numFeatures)
        var member = "["
        rawMembership.foreach { x => member += x + "," }
        member = member.dropRight(1) + "]"
        var outString = "[-1.0," + features + member + "]"
        outString
      }.coalesce(1, true) saveAsTextFile (pathTrain.replace("tra.data", "tra-LHS-Memb-map" + numPartition + "k" + K + ".data"))

    }else if (stage == "classification"){
      //Second stage. Predict with the info of the class membership computed
      timeMemberBeg = System.nanoTime

      // Generate the schema
      val label = Array(StructField("label", DoubleType, true))
      val schemaLabel = StructType(label)
      val schemaFeatures = SchemaUtils.appendColumn(schemaLabel, "features", new VectorUDT)
      val schema = SchemaUtils.appendColumn(schemaFeatures, "rawPrediction", new VectorUDT)
      val trainWithMembership = sqlContext.createDataFrame(sc.textFile(pathTrain: String, numPartition).map { line =>
        val rawArray = line.replace(",[", ",,").replace("[", "").replace("]", "").split(",,")

        val rawLabel = Row(rawArray(0).toDouble)
        val rawFeatures = rawArray(1).split(",").map { x => x.toDouble }
        val rawMembership = rawArray(2).split(",").map { x => x.toDouble }

        val features = new ArrayBuffer[Any]
        features.append(Vectors.dense(rawFeatures))
        val membership = new ArrayBuffer[Any]
        membership.append(Vectors.dense(rawMembership))
        val row = Row.fromSeq(rawLabel.toSeq ++ features ++ membership)
        row
      }, schema).persist
      val test = sc.textFile(pathTest: String, numPartition).map(line => converter.parserToLabeledPoint(line)).toDF().persist
      val knnFuzzy = new HS_FkNNClassifier().setTopTreeSize(trainWithMembership.count.toInt / 500).setK(K)
      val knnModelFuzzy = knnFuzzy.fitFuzzy(trainWithMembership, numClasses)
      val labelAndPredicted = knnModelFuzzy.transformPredict(test)
      labelAndPredicted.count
      timeEnd = System.nanoTime
      timePredictEnd = timeEnd

      // Write the output
      val metrics = new MulticlassMetrics(labelAndPredicted)
      val accuracy = metrics.accuracy
      val cm = metrics.confusionMatrix
      val binaryMetrics = new BinaryClassificationMetrics(labelAndPredicted)
      val AUC = binaryMetrics.areaUnderROC.toString()

      var writerReport = new ListBuffer[String]
      writerReport += "***Report.txt ==> Contain: Confusion Matrix; Precision; Total Runtime***\n"
      writerReport += "@ConfusionMatrix\n" + cm
      writerReport += "\n@Accuracy\n" + accuracy
      writerReport += "\n@AUC\n" + AUC
      writerReport += "\n@PredictionRuntime\n" + (timePredictEnd - timeMemberBeg) / 1e9
      val Report = sc.parallelize(writerReport, 1)
      Report.saveAsTextFile(pathOutput + "/Report.txt")
    }else{
      logger.error("=> wrong parameter stage:")
      System.err.println("Options: GAHS-Memb or LHS-Memb or classification")
      System.exit(1)
    }

  }
}
