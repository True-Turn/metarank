package ai.metarank.main

import ai.metarank.main.command.train.SplitStrategy
import ai.metarank.main.command.train.SplitStrategy.{FieldStrategy, HoldLastStrategy, RandomSplit, TimeSplit}
import ai.metarank.model.Field.StringField
import ai.metarank.model.Identifier.UserId
import ai.metarank.model.{QueryMetadata, Timestamp}
import cats.effect.unsafe.implicits.global
import io.github.metarank.ltrlib.model.{DatasetDescriptor, LabeledItem, Query}
import io.github.metarank.ltrlib.model.Feature.SingularFeature
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SplitStrategyTest extends AnyFlatSpec with Matchers {
  it should "parse inputs" in {
    SplitStrategy.parse("random=10%") shouldBe Right(RandomSplit(10))
    SplitStrategy.parse("random") shouldBe Right(RandomSplit(80))
    SplitStrategy.parse("field=split:train:test") shouldBe Right(FieldStrategy("split", "train", "test"))
  }

  val desc  = DatasetDescriptor(List(SingularFeature("foo")))
  val now   = Timestamp.now
  val query = QueryMetadata(Query(desc, List(LabeledItem(1.0, 1, Array(1.0)))), now, None, Nil)

  "time-split" should "handle unbalanced small inputs, size=1" in {
    val split = TimeSplit(80).split(desc, List(query, query)).unsafeRunSync()
    split.test.groups.size shouldBe 1
    split.train.groups.size shouldBe 1
  }

  it should "handle unbalanced small inputs, size=2" in {
    val split = TimeSplit(80).split(desc, List(query, query)).unsafeRunSync()
    split.test.groups.size shouldBe 1
    split.train.groups.size shouldBe 1
  }

  it should "handle unbalanced small inputs, size=3" in {
    val split = TimeSplit(80).split(desc, List(query, query, query)).unsafeRunSync()
    split.test.groups.size shouldBe 1
    split.train.groups.size shouldBe 2
  }

  "field split" should "split by field value" in {
    val result = FieldStrategy("split", "train", "test")
      .split(
        desc,
        List(
          query.copy(fields = List(StringField("split", "train"))),
          query.copy(fields = List(StringField("split", "test")))
        )
      )
      .unsafeRunSync()
    result.test.groups.size shouldBe 1
    result.train.groups.size shouldBe 1
  }

  "hold-last split" should "hold back the latest rankings of each user" in {
    // Group ids number each user's rankings in time order: 0..9 for u1, 100..109 for u2
    val queries = for {
      (user, offset) <- List("u1" -> 0, "u2" -> 100)
      i              <- 0 until 10
    } yield {
      QueryMetadata(Query(offset + i, Array(1.0), Array(1.0)), Timestamp(1000L * i), Some(UserId(user)), Nil)
    }
    val split = HoldLastStrategy(80).split(desc, queries).unsafeRunSync()
    split.test.groups.map(_.group).sorted shouldBe List(8, 9, 108, 109)
    split.train.groups.size shouldBe 16
  }
}
