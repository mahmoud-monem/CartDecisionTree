import pandas as pd
import numpy as np


class Node:
    def __init__(self, label=str()):
        self.label = label
        self.edges = dict()

    def addEdge(self, edgeLabel: str, nodeLabel=str()):
        self.edges[edgeLabel] = Node(nodeLabel)

    def setLabel(self, label: str):
        self.label = label


class CartDecisionTree:
    def __init__(self, data: pd.DataFrame, targetVariable="Decision"):
        self.data = data
        self.targetVariable = targetVariable
        self.root = Node()
        self.classes = set()
        for idx in range(len(self.data)):
            self.classes.add(self.data[targetVariable][idx])

    def fit(self):
        indices = np.arange(0, len(self.data))

        unUsedAttributes = list()
        for attribute in self.data:
            if attribute == self.targetVariable or attribute == "id":
                continue
            unUsedAttributes.append(attribute)

        self.build(self.root, indices, unUsedAttributes)

    def build(self, node: Node, indices: np.ndarray, unUsedAttributes: list):
        nodeLabel, attributeValues = self.calculateGiniIndex(indices, unUsedAttributes)
        node.setLabel(nodeLabel)
        # print(nodeLabel)

        for value in attributeValues:
            flag = False
            # print(attributeValues[value][1], attributeValues[value][0], attributeValues[value]["total"])
            for classLabel in self.classes:
                if attributeValues[value][classLabel] == attributeValues[value]["total"]:
                    node.addEdge(value, classLabel)
                    flag = True
                    break

            if flag:
                continue

            node.addEdge(value)
            newUnUsed = list()
            newIndices = np.array([])
            idx2 = 0
            for idx in indices:
                if idx2 < len(attributeValues[value]["indices"]):
                    if idx == attributeValues[value]["indices"][idx2]:
                        idx2 += 1
                        continue
                    newIndices = np.append(newIndices, [idx])
            for attribute in unUsedAttributes:
                if attribute == nodeLabel:
                    continue
                newUnUsed.append(attribute)
            self.build(node.edges[value], attributeValues[value]["indices"], newUnUsed)

    def calculateGiniIndex(self, indices: np.ndarray, unUsedAttributes: list) -> str:
        minIndexLabel = str()
        minIndexAttributeValues = dict()
        minIndexValue = 1

        for attribute in unUsedAttributes:
            attributeValues = dict()
            for idx in indices:
                val = self.data[attribute][idx]
                if attributeValues.get(val):
                    attributeValues[val]["total"] += 1
                    attributeValues[val][self.data[self.targetVariable][idx]] += 1
                    attributeValues[val]["indices"] = np.append(attributeValues[val]["indices"], idx)
                else:
                    attributeValues[val] = dict()
                    attributeValues[val]["total"] = 1
                    attributeValues[val]["indices"] = np.array([idx])
                    for classLabel in self.classes:
                        attributeValues[val][classLabel] = 0
                    attributeValues[val][self.data[self.targetVariable][idx]] = 1

            giniIndex = 0
            for value in attributeValues:
                temp = 1
                tot = attributeValues[value]["total"]
                for classLabel in self.classes:
                    temp -= (attributeValues[value][classLabel] / tot) ** 2
                giniIndex += (tot / len(indices)) * temp

            if giniIndex < minIndexValue:
                minIndexValue = giniIndex
                minIndexLabel = attribute
                minIndexAttributeValues = attributeValues

        return minIndexLabel, minIndexAttributeValues

    def predict(self, testData: pd.DataFrame) -> np.ndarray:
        predicted = np.array([])
        for idx in range(len(testData)):
            self._predict(self.root, testData.loc[idx])
        return predicted

    def _predict(self, node: Node, testData):
        if not node.edges:
            return node.label
        else:
            for edge in node.edges:
                return self._predict(edge, testData)
