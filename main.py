from src.geminiSetGenerator import process_questions
from src.utils import loadData
import os

PATH_GOLDENSET = os.path.join("data", "stackoverflow_dataset.json")

rawgolden = loadData(PATH_GOLDENSET, default_type=list)

if rawgolden:
    process_questions(rawgolden)   