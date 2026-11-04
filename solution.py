#!/usr/bin/env python3

import argparse
import csv
import math
import os
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def read_queries(path):
    result = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            query_id, text = line.rstrip("\n").split("\t", 1)
            result[int(query_id)] = text
    return result


def read_qrels(path):
    result = defaultdict(dict)
    with open(path, encoding="utf-8") as file:
        for line in file:
            query_id, _, document_id, label = line.split()
            result[int(query_id)][document_id] = int(label)
    return dict(result)


def read_documents(path):
    document_ids = []
    urls = []
    titles = []
    bodies = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\n").split("\t", 3)
            fields += [""] * (4 - len(fields))
            document_id, url, title, body = fields[:4]
            document_ids.append(document_id)
            urls.append(url)
            titles.append(title)
            bodies.append(body)
    return document_ids, urls, titles, bodies


def read_objects(path):
    with open(path, encoding="utf-8") as file:
        return [(int(row["ObjectId"]), int(row["QueryId"]), row["DocumentId"])
                for row in csv.DictReader(file)]


def ndcg_at_10(labels):
    ordered = list(labels[:10])
    dcg = sum((2.0 ** label - 1.0) / math.log2(position + 2)
              for position, label in enumerate(ordered))
    ideal = sorted(labels, reverse=True)[:10]
    idcg = sum((2.0 ** label - 1.0) / math.log2(position + 2)
               for position, label in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(scores, qrels):
    values = []
    for query_id, labels in qrels.items():
        ranked = sorted(labels, key=lambda document_id: (-scores[query_id].get(document_id, 0.0), document_id))
        values.append(ndcg_at_10([labels[document_id] for document_id in ranked]))
    return float(np.mean(values)) if values else 0.0


class BM25F:
    def __init__(self, vectorizer, title_counts, body_counts, title_lengths, body_lengths, idf,
                 title_weight=3.0, body_weight=1.0, k1=1.2, title_b=0.3, body_b=0.75):
        self.vectorizer = vectorizer
        self.title_counts = title_counts
        self.body_counts = body_counts
        self.title_lengths = np.asarray(title_lengths, dtype=np.float32)
        self.body_lengths = np.asarray(body_lengths, dtype=np.float32)
        self.avg_title_length = max(float(self.title_lengths.mean()), 1.0)
        self.avg_body_length = max(float(self.body_lengths.mean()), 1.0)
        self.idf = idf.astype(np.float32)
        self.title_weight = float(title_weight)
        self.body_weight = float(body_weight)
        self.k1 = float(k1)
        self.title_b = float(title_b)
        self.body_b = float(body_b)
        self.analyzer = vectorizer.build_analyzer()

    def query_terms(self, query):
        indices = []
        for token in self.analyzer(query):
            index = self.vectorizer.vocabulary_.get(token)
            if index is not None:
                indices.append(index)
        return np.unique(np.asarray(indices, dtype=np.int32)) if indices else np.empty(0, dtype=np.int32)

    def score_indices(self, query, document_indices):
        terms = self.query_terms(query)
        if terms.size == 0 or len(document_indices) == 0:
            return np.zeros(len(document_indices), dtype=np.float32)
        title = self.title_counts[document_indices][:, terms].toarray().astype(np.float32, copy=False)
        body = self.body_counts[document_indices][:, terms].toarray().astype(np.float32, copy=False)
        title_denominator = 1.0 - self.title_b + self.title_b * self.title_lengths[document_indices] / self.avg_title_length
        body_denominator = 1.0 - self.body_b + self.body_b * self.body_lengths[document_indices] / self.avg_body_length
        title_denominator = np.maximum(title_denominator, 1e-6)
        body_denominator = np.maximum(body_denominator, 1e-6)
        weighted_tf = self.title_weight * title / title_denominator[:, None]
        weighted_tf += self.body_weight * body / body_denominator[:, None]
        numerator = weighted_tf * (self.k1 + 1.0)
        denominator = weighted_tf + self.k1
        return np.sum(self.idf[terms][None, :] * numerator / denominator, axis=1, dtype=np.float32)

    def score_pairs(self, queries, pairs, document_index):
        grouped = defaultdict(list)
        for query_id, document_id, position in pairs:
            index = document_index.get(document_id)
            if index is not None:
                grouped[query_id].append((position, index))
        result = {}
        for query_id, values in grouped.items():
            positions = [item[0] for item in values]
            indices = np.asarray([item[1] for item in values], dtype=np.int32)
            scores = self.score_indices(queries[query_id], indices)
            result[query_id] = {pairs_position: float(score) for pairs_position, score in zip(positions, scores)}
        return result


def build_model(data_dir, title_weight, body_weight, k1, title_b, body_b):
    document_ids, _, titles, bodies = read_documents(os.path.join(data_dir, "vkmarco-docs.tsv"))
    vectorizer = CountVectorizer(dtype=np.float32, lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    vectorizer.fit(title + " " + body for title, body in zip(titles, bodies))
    title_counts = vectorizer.transform(titles).tocsr()
    body_counts = vectorizer.transform(bodies).tocsr()
    document_frequency = np.asarray(((title_counts > 0) + (body_counts > 0)).astype(bool).sum(axis=0)).ravel()
    document_frequency = np.maximum(document_frequency, 1.0)
    document_count = len(document_ids)
    idf = np.log1p((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
    model = BM25F(
        vectorizer,
        title_counts,
        body_counts,
        np.asarray(title_counts.sum(axis=1)).ravel(),
        np.asarray(body_counts.sum(axis=1)).ravel(),
        idf,
        title_weight=title_weight,
        body_weight=body_weight,
        k1=k1,
        title_b=title_b,
        body_b=body_b,
    )
    return model, document_ids


def score_query_documents(model, queries, query_documents, document_index):
    scores = {}
    for query_id, document_ids in query_documents.items():
        indices = np.asarray([document_index[document_id] for document_id in document_ids], dtype=np.int32)
        values = model.score_indices(queries[query_id], indices)
        scores[query_id] = dict(zip(document_ids, values.astype(float)))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_file", required=True)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    start = timer()

    train_queries = read_queries(os.path.join(args.data_dir, "vkmarco-doctrain-queries.tsv"))
    train_qrels = read_qrels(os.path.join(args.data_dir, "vkmarco-doctrain-qrels.tsv"))
    test_queries = read_queries(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv"))
    objects = read_objects(os.path.join(args.data_dir, "objects.csv"))

    model, document_ids = build_model(args.data_dir, title_weight=2.5, body_weight=1.0,
                                      k1=1.2, title_b=0.5, body_b=0.75)
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}

    train_documents = {query_id: list(labels) for query_id, labels in train_qrels.items()}
    train_scores = score_query_documents(model, train_queries, train_documents, document_index)
    train_ndcg = evaluate(train_scores, train_qrels)
    print(f"train_ndcg@10={train_ndcg:.6f}")

    grouped_test = defaultdict(list)
    for object_id, query_id, document_id in objects:
        grouped_test[query_id].append((object_id, document_id))
    with open(args.submission_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ObjectId", "Score"])
        for query_id, values in grouped_test.items():
            document_ids_for_query = [document_id for _, document_id in values]
            indices = np.asarray([document_index[document_id] for document_id in document_ids_for_query], dtype=np.int32)
            scores = model.score_indices(test_queries[query_id], indices)
            for (object_id, _), score in zip(values, scores):
                writer.writerow([object_id, f"{float(score):.8f}"])
    print(f"finished, elapsed={timer() - start:.3f}")


if __name__ == "__main__":
    main()
