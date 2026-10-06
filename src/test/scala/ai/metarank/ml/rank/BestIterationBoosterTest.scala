package ai.metarank.ml.rank

import io.github.metarank.lightgbm4j.LGBMBooster.FeatureImportanceType
import io.github.metarank.ltrlib.booster.{LightGBMBooster, LightGBMOptions, XGBoostBooster, XGBoostOptions}
import io.github.metarank.ltrlib.model.Feature.{CategoryFeature, SingularFeature}
import io.github.metarank.ltrlib.model.{Dataset, DatasetDescriptor, Query}
import io.github.metarank.ltrlib.ranking.pairwise.LambdaMART
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class BestIterationBoosterTest extends AnyFlatSpec with Matchers {
  val desc  = DatasetDescriptor(List(SingularFeature("f")))
  val items = 10

  // Train rewards high feature values and test rewards low ones, so test NDCG is best after the first tree
  def dataset(seed: Int, relevantIfHigh: Boolean): Dataset = {
    val random = new Random(seed)
    val queries = (0 until 100).map { group =>
      val values = Array.fill(items)(random.nextDouble())
      val sorted = values.sorted
      val labels = values.map(v => if (if (relevantIfHigh) v >= sorted(items - 3) else v <= sorted(2)) 1.0 else 0.0)
      Query(group, labels, values)
    }
    Dataset(desc, queries.toList)
  }

  val train = dataset(1, relevantIfHigh = true)
  val test  = dataset(2, relevantIfHigh = false)

  val lgbmOptions = LightGBMOptions(
    trees = 100,
    learningRate = 0.1,
    ndcgCutoff = 10,
    maxDepth = 8,
    randomSeed = 0,
    numLeaves = 16,
    featureFraction = 1.0,
    earlyStopping = Some(20),
    debias = false
  )
  val xgbOptions = XGBoostOptions(
    trees = 100,
    learningRate = 0.1,
    ndcgCutoff = 10,
    maxDepth = 8,
    randomSeed = 0,
    subsample = 1.0,
    earlyStopping = Some(20),
    treeMethod = "exact",
    debias = false
  )

  def lgbmTrees(b: LightGBMBooster): Int =
    "(?m)^Tree=".r.findAllIn(b.model.saveModelToString(0, 0, FeatureImportanceType.SPLIT)).size
  def xgbTrees(b: XGBoostBooster): Int = b.model.getModelDump(null: String, false).length

  "LightGBM" should "keep only the trees up to the best test iteration" in {
    lgbmTrees(LambdaMART(train, LightGBMBooster, Some(test), lgbmOptions).fit(lgbmOptions)) should be > 20
    lgbmTrees(LambdaMART(train, BestIterationBooster.LightGBM, Some(test), lgbmOptions).fit(lgbmOptions)) shouldBe 1
  }

  "XGBoost" should "keep only the trees up to the best test iteration" in {
    xgbTrees(LambdaMART(train, XGBoostBooster, Some(test), xgbOptions).fit(xgbOptions)) should be > 20
    xgbTrees(LambdaMART(train, BestIterationBooster.XGBoost, Some(test), xgbOptions).fit(xgbOptions)) shouldBe 1
  }

  it should "save the best iteration when it holds categorical splits" in {
    val catDesc = DatasetDescriptor(List(CategoryFeature("c")))
    def catDataset(seed: Int, relevant: Set[Int]): Dataset = {
      val random = new Random(seed)
      val queries = (0 until 100).map { group =>
        val values = Array.fill(items)(random.nextInt(20).toDouble)
        Query(group, values.map(v => if (relevant.contains(v.toInt)) 1.0 else 0.0), values)
      }
      Dataset(catDesc, queries.toList)
    }
    val opts = xgbOptions.copy(treeMethod = "hist")
    val booster =
      LambdaMART(catDataset(1, Set(3, 11)), BestIterationBooster.XGBoost, Some(catDataset(2, Set(5, 17))), opts)
        .fit(opts)
    // Tree dumps fail on categorical splits, so trees are counted in the saved JSON model
    def trees(b: XGBoostBooster) =
      "\"num_trees\":\"(\\d+)\"".r.findFirstMatchIn(new String(b.save())).map(_.group(1).toInt)
    trees(booster).get should be < opts.trees
    trees(XGBoostBooster(booster.save())) shouldBe trees(booster)
  }
}
