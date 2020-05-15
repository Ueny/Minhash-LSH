from pyspark import SparkConf, SparkContext
import sys
import json
from itertools import combinations
import time

input_file = 'train_review.json'
output_file = 'task1.res'

# minhash and lSH parameters
n_hashes = 40
b = 40
jac_sim = 0.05


def init_user_dict(user_id):
    user_dict = dict()
    index = 0
    for uid in user_id:
        user_dict[uid] = index
        index += 1
    return user_dict

def minhash(iterator, n_hashes, n_buckets):
    for key, values in iterator:
        signature = []
        for a in range(1, n_hashes+1):
            min_value = n_buckets
            for value in values:
                # (2749*value+5323+a*((937*value+4093)%1433))%n_buckets
                min_value = min(min_value, ((49+a)*value+323)%n_buckets)
            signature.append(min_value)
        yield (key, signature)

def lsh(iterator, n_hashes, n_bands):
    r = n_hashes/n_bands

    for key, signature in iterator:
        for i in range(n_bands):
            sig_band = signature[int(i*r):int((i+1)*r)]
            value = ''
            for sig in sig_band:
                value += str(sig)
            yield ((i, value), key)

def jaccard(iterator, rv_dict):
    for b1, b2 in iterator:
        u1_list = rv_dict[b1]
        u2_list = rv_dict[b2]
        inter = set(u1_list) & set(u2_list)
        union = set(u1_list) | set(u2_list)
        sim = len(inter)/len(union)
        if sim >= jac_sim:
            yield (b1, b2, sim)

start_time = time.time()
conf = SparkConf().setAppName('inf553_hw3_task1').setMaster('local[*]').set('spark.driver.memory', '4G')
sc = SparkContext(conf=conf)

review = sc.textFile(input_file)\
    .map(lambda s: json.loads(s))\
    .persist()

user_id = review.map(lambda s: s['user_id'])\
    .distinct()\
    .sortBy(lambda s: s, ascending=True)\
    .collect()

user_dict = init_user_dict(user_id)

biz_usrs = review.map(lambda s: (s['business_id'], s['user_id']))\
    .distinct()\
    .map(lambda s: (s[0], user_dict[s[1]]))\
    .groupByKey()\
    .mapValues(list)\
    .persist()

review_dict = dict()
for key, value in biz_usrs.collect():
    review_dict[key] = value

candidates = biz_usrs.mapPartitions(lambda s: minhash(iterator=s, n_hashes=n_hashes, n_buckets=len(user_dict.keys())))\
    .mapPartitions(lambda s: lsh(iterator=s, n_hashes=n_hashes, n_bands=b)) \
    .groupByKey() \
    .mapValues(list) \
    .filter(lambda s: len(s[1]) > 1)\
    .flatMap(lambda s: list(combinations(sorted(s[1]), 2))) \
    .distinct() \
    .mapPartitions(lambda s: jaccard(iterator=s, rv_dict=review_dict))\
    .persist()

result = candidates.map(lambda s: dict({'b1': s[0], 'b2': s[1], 'sim': s[2]}))\
    .collect()
candidates = candidates.collect()

ground_truth = review.map(lambda s: (s['user_id'], s['business_id']))\
    .distinct()\
    .groupByKey()\
    .mapValues(list)\
    .filter(lambda s: len(s[1])>1)\
    .flatMap(lambda s: list(combinations(sorted(s[1]), 2)))\
    .distinct()\
    .mapPartitions(lambda s: jaccard(iterator=s, rv_dict=review_dict))\
    .collect()

with open(output_file, 'w') as f:
    for piece in result:
        f.write(json.dumps(piece) + '\n')

numerator = len(set(candidates) & set(ground_truth))
denominator = len(set(candidates) | set(ground_truth))
print('Accuracy: ', numerator/denominator)
print('Duration: ', time.time()-start_time)