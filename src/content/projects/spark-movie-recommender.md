---
title: "Large-Scale Movie Recommender — Spark + Scala"
description: "Distributed recommender in Scala 3.3 + Spark 3.5. ALS collaborative filtering, LSH/MinHash similarity search, biased baseline predictor across a 25-partition HashPartitioner."
pubDate: 2024-05-25
tags: ["Spark", "Scala", "ALS", "distributed"]
---

## What this is

Distributed movie recommender on Apache Spark in Scala 3, packaged as one `sbt assembly` jar for cluster submission.

## How it works

Four modules in the jar:

- **Analytics**: yearly and all-time popularity, top-genre breakdowns. Spark aggregations partitioned 25-way through a custom `HashPartitioner` to balance shuffles.
- **Baseline predictor**: per-user and per-movie bias terms, normalized into [1, 5]. Reference for the other predictors.
- **Collaborative filtering**: ALS, 20 iterations.
- **Similarity search**: LSH with MinHash for fast approximate nearest-neighbor lookup of similar users.

JUnit tests on each stage. JDK 11, Scala 3.3.1, Spark 3.5.0.

CS-460 Systems for Data Management & DS at EPFL.
