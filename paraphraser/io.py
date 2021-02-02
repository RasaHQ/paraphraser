from typing import Text, List, Any
import csv
from pathlib import Path

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.loading import load_data
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging

logger = logging.getLogger()


def read_from_csv(file_path: Text) -> List[Text]:

    collection = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, quotechar='"')
        for line in reader:
            collection.append(line)
    return collection


def read_collection_from_csv(file_path: Text) -> List[Message]:

    collection = read_from_csv(file_path)
    all_sentences = []
    for line in collection:
        if len(line) == 2:
            sentence, label = line[0], line[1]
            all_sentences.append(Message.build(text=sentence, intent=label))
        elif len(line) == 1:
            sentence = line[0]
            all_sentences.append(Message.build(text=sentence))
        else:
            raise RuntimeError("Input CSV file does not adhere to the correct format")
    return all_sentences


def read_rasa_collection(file_path: Text) -> List[Message]:

    data = load_data(file_path)
    return data.nlu_examples

def read_collection(data_directory: Text, file_path: Text) -> Message:

    input_file_path = Path(data_directory) / file_path
    if str(input_file_path).endswith("csv"):
        logger.debug("Loading as a csv file")
        return read_collection_from_csv(input_file_path)
    else:
        logger.debug("Loading as a Rasa supported NLU data file")
        try:
            return read_rasa_collection(input_file_path)
        except Exception as e:
            raise RuntimeError("Input file not in the supported format. Please refer to the README to check the supported formats.")


def dump_to_yaml(collection: List[Message], file_path: Text) -> None:

    data = TrainingData(collection)
    data.persist_nlu(file_path)

def dump_to_csv(csv_lines: List[List[Text]], file_path: Text) -> None:

    with open(file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csvwriter.writerow(line)


def serialize_collection_as_csv(collection: List[Message]) -> List[List[Text]]:

    csv_lines = []
    for message in collection:
        text = message.get("text")
        intent = message.get("intent")
        paraphrases = message.get("metadata").get("paraphrases")

        for paraphrase in paraphrases:
            csv_lines.append([text, intent, paraphrase])

    return csv_lines


def write_collection(collection: List[Message], output_directory: Text, format: Text):

    output_file_path = Path(output_directory) / f"augmented_data.{format}"

    if format == "csv":
        csv_content = serialize_collection_as_csv(collection)
        dump_to_csv(csv_content, output_file_path)
    elif format == "yaml" or format == "yml":
        dump_to_yaml(collection, output_file_path)
