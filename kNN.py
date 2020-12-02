from typing import Dict, List, NamedTuple
import csv
from collections import defaultdict

from scratch.linear_algebra import Vector, distance

class LabeledPoint(NamedTuple):
    point:Vector
    label:str

import requests

data = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

with open('iris.data', 'w') as f:
    f.write(data.text)

def parse_iris_row(row):
    '''
    꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비, 분류
    '''
    measurements = [float(value) for value in row[:-1]]
    label = row[-1]#.split("-")[-1]

    return LabeledPoint(measurements, label)


with open('iris.data') as f:
    reader = csv.reader(f)
    iris_data = []
    for row in reader:
        if row:
            iris_data.append(parse_iris_row(row))



# 데이터를 살펴보기 위해 종/레이블로 무리 지어보자.
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

from matplotlib import pyplot as plt
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x'] # 3개의 분류가 있으므로 3개의 마커가 필요하다.

fig, ax = plt.subplots(2, 3)
for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc='lower right', prop={'size':6})
plt.show()

from collections import Counter

def majority_vote(labels: List[str]) -> str:
    """labels는 가장 가까운 데이터부터 가장 먼 데이터 순서로 정렬되어 있다고 가정"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    # 레이블된 포인트를 가장 가까운 데이터부터 가장 먼 데이터 순서로 정렬
    by_distance = sorted(labeled_points, key=lambda lp:distance(lp.point, new_point))

    # 가장 가까운 k 데이터 포인트의 레이블을 살펴보고
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # 투표한다.
    return majority_vote(k_nearest_labels)


import random
from scratch.machine_learning import split_data

random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

from typing import Tuple

confusion_matrix : Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)

# 차원의 저주

def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]

import tqdm
dimentions = range(1, 101)
avg_distances = []
mis_distances = []

for dim in tqdm.tqdm(dimentions, desc="Curse of Dimentionality"):
    distances = random_distances(dim, 10000)
    avg_distances.append(sum(distances) / 10000)
    min_distances.append(min(distances))
