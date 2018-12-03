#!/bin/bash

/opt/spark-2.3.2-bin-hadoop2.7/bin/spark-submit --master local[*] --class org.apache.spark.run.runHS_FkNN ./target/HS-FkNN-1.0.jar ./dataset/page-blocks.header ./dataset/page-blocks-tra.data ./dataset/page-blocks-tst.data 3 10 LHS-Memb ./LHS-Memb

/opt/spark-2.3.2-bin-hadoop2.7/bin/spark-submit --master local[*] --class org.apache.spark.run.runHS_FkNN ./target/HS-FkNN-1.0.jar ./dataset/page-blocks.header ./dataset/page-blocks-tra-LHS-Memb-map10k3.data/part-00000 ./dataset/page-blocks-tst.data 3 10 classification ./LHS-Class

/opt/spark-2.3.2-bin-hadoop2.7/bin/spark-submit --master local[*] --class org.apache.spark.run.runHS_FkNN ./target/HS-FkNN-1.0.jar ./dataset/page-blocks.header ./dataset/page-blocks-tra.data ./dataset/page-blocks-tst.data 3 10 GAHS-Memb ./GAHS-Memb

/opt/spark-2.3.2-bin-hadoop2.7/bin/spark-submit --master local[*] --class org.apache.spark.run.runHS_FkNN ./target/HS-FkNN-1.0.jar ./dataset/page-blocks.header ./dataset/page-blocks-tra-GAHS-Memb-map10k3.data/part-00000 ./dataset/page-blocks-tst.data 3 10 classification ./GAHS-Class
