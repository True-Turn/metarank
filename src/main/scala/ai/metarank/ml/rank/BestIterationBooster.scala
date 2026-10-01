package ai.metarank.ml.rank

import ai.metarank.util.Logging
import io.github.metarank.lightgbm4j.LGBMBooster.FeatureImportanceType
import io.github.metarank.lightgbm4j.{LGBMBooster, LGBMDataset}
import io.github.metarank.ltrlib.booster.Booster.{BoosterFactory, DatasetOptions}
import io.github.metarank.ltrlib.booster.{
  BoosterDataset,
  LightGBMBooster,
  LightGBMOptions,
  XGBoostBooster,
  XGBoostOptions
}
import ml.dmlc.xgboost4j.java.DMatrix

import java.io.ByteArrayInputStream
import scala.jdk.CollectionConverters.*

// Boosters that keep the iteration with the best test NDCG when early stopping fires, not the last one
object BestIterationBooster {

  object LightGBM extends BoosterFactory[LGBMDataset, LightGBMBooster, LightGBMOptions] with Logging {
    override def apply(string: Array[Byte]): LightGBMBooster = LightGBMBooster(string)
    override def formatData(ds: BoosterDataset, parent: Option[LGBMDataset], options: LightGBMOptions): LGBMDataset =
      LightGBMBooster.formatData(ds, parent, options)
    override def closeData(d: LGBMDataset): Unit = LightGBMBooster.closeData(d)

    override def train(
        dataset: LGBMDataset,
        test: Option[LGBMDataset],
        options: LightGBMOptions,
        dso: DatasetOptions
    ): LightGBMBooster = {
      val params = Map(
        "objective"                   -> "lambdarank",
        "metric"                      -> "ndcg",
        "lambdarank_truncation_level" -> options.ndcgCutoff.toString,
        "max_depth"                   -> options.maxDepth.toString,
        "learning_rate"               -> options.learningRate.toString,
        "num_leaves"                  -> options.numLeaves.toString,
        "seed"                        -> options.randomSeed.toString,
        "categorical_feature"         -> dso.categoryFeatures.mkString(","),
        "feature_fraction"            -> options.featureFraction.toString,
        "eval_at"                     -> options.ndcgCutoff.toString
      ).map(kv => s"${kv._1}=${kv._2}").mkString(" ")
      val model = LGBMBooster.create(dataset, params)
      test.foreach(t => model.addValidData(t))
      var it       = 0
      var best     = 0.0
      var bestIter = 0
      var stop     = false
      while (it < options.trees && !stop) {
        it += 1
        model.updateOneIter()
        val ndcgTrain = model.getEval(0)(0)
        test match {
          case Some(_) =>
            val ndcgTest = model.getEval(1)(0)
            logger.info(s"[$it] NDCG@train = $ndcgTrain NDCG@test = $ndcgTest")
            if (ndcgTest > best) {
              best = ndcgTest
              bestIter = it
            }
            options.earlyStopping.foreach(threshold =>
              if (it - bestIter > threshold) {
                logger.info(s"early stop: $threshold rounds passed, best=$best at [$bestIter], last=$ndcgTest")
                stop = true
              }
            )
          case None =>
            logger.info(s"[$it] NDCG@train = $ndcgTrain")
        }
      }
      if (test.isDefined && bestIter > 0 && bestIter < it) {
        // Saving only the first bestIter trees is the only way to truncate a LightGBM model
        val truncated = model.saveModelToString(0, bestIter, FeatureImportanceType.SPLIT)
        model.close()
        logger.info(s"keeping the model as of iteration $bestIter of $it")
        LightGBMBooster(LGBMBooster.loadModelFromString(truncated))
      } else {
        LightGBMBooster(model)
      }
    }
  }

  object XGBoost extends BoosterFactory[DMatrix, XGBoostBooster, XGBoostOptions] with Logging {
    override def apply(string: Array[Byte]): XGBoostBooster = XGBoostBooster(string)
    override def formatData(ds: BoosterDataset, parent: Option[DMatrix], options: XGBoostOptions): DMatrix =
      XGBoostBooster.formatData(ds, parent, options)
    override def closeData(d: DMatrix): Unit = XGBoostBooster.closeData(d)

    override def train(
        dataset: DMatrix,
        test: Option[DMatrix],
        options: XGBoostOptions,
        dso: DatasetOptions
    ): XGBoostBooster = {
      val opts = Map[String, Object](
        "objective"           -> "rank:pairwise",
        "eval_metric"         -> s"ndcg@${options.ndcgCutoff}",
        "num_round"           -> Integer.valueOf(options.trees),
        "max_depth"           -> options.maxDepth.toString,
        "eta"                 -> options.learningRate.toString,
        "seed"                -> options.randomSeed.toString,
        "subsample"           -> options.subsample.toString,
        "tree_method"         -> options.treeMethod,
        "enable_categorical"  -> (if (dso.categoryFeatures.isEmpty) "false" else "true"),
        "lambdarank_unbiased" -> (if (options.debias) "true" else "false")
      ).asJava
      val model    = ml.dmlc.xgboost4j.java.XGBoost.train(dataset, opts, 0, Map.empty.asJava, null, null)
      var it       = 0
      var best     = 0.0
      var bestIter = 0
      var bestSnapshot: Array[Byte] = null
      var stop                      = false
      while (it < options.trees && !stop) {
        model.update(dataset, it)
        val ndcgTrain = evalMetric(model, dataset, it)
        test match {
          case Some(t) =>
            val ndcgTest = evalMetric(model, t, it)
            logger.info(
              s"[$it] NDCG@${options.ndcgCutoff}:train = $ndcgTrain NDCG@${options.ndcgCutoff}:test = $ndcgTest"
            )
            if (ndcgTest > best) {
              best = ndcgTest
              bestIter = it
              // This XGBoost build cannot slice a booster, so the best one is kept as a serialised copy
              bestSnapshot = model.toByteArray()
            }
            options.earlyStopping.foreach(threshold =>
              if (it - bestIter > threshold) {
                logger.info(s"early stop: $threshold rounds passed, best=$best at [$bestIter], last=$ndcgTest")
                stop = true
              }
            )
          case None =>
            logger.info(s"[$it] NDCG@train = $ndcgTrain")
        }
        it += 1
      }
      val featureTypes = (0 until dso.dims).map(x => if (dso.categoryFeatures.contains(x)) "c" else "q").toArray
      if (bestSnapshot != null && bestIter < it - 1) {
        model.dispose()
        logger.info(s"keeping the model as of iteration ${bestIter + 1} of $it")
        XGBoostBooster(ml.dmlc.xgboost4j.java.XGBoost.loadModel(new ByteArrayInputStream(bestSnapshot)), featureTypes)
      } else {
        XGBoostBooster(model, featureTypes)
      }
    }

    private def evalMetric(model: ml.dmlc.xgboost4j.java.Booster, dataset: DMatrix, it: Int): Double =
      model.evalSet(Array(dataset), Array("test"), it).split(':').last.toDouble
  }
}
