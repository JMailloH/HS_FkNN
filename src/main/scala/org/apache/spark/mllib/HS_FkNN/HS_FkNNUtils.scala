package org.apache.spark.mllib.HS_FkNN

import org.apache.spark.ml.{linalg => newlinalg}
import org.apache.spark.mllib.{linalg => oldlinalg}
import org.apache.spark.mllib.util.MLUtils

object HS_FkNNUtils {

  import oldlinalg.VectorImplicits._

  def fastSquaredDistance(
                           v1: newlinalg.Vector,
                           norm1: Double,
                           v2: newlinalg.Vector,
                           norm2: Double,
                           precision: Double = 1e-6): Double = {
    MLUtils.fastSquaredDistance(v1, norm1, v2, norm2, precision)
  }

}
