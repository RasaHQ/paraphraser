from typing import Text, List, Any
import csv
from pathlib import Path

from rasa.shared.nlu.training_data.message import Message
from collections import OrderedDict


def read_from_csv(file_path: Text) -> List[Text]:

    collection = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
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


def read_collection(data_directory: Text, file_path: Text) -> Message:

    input_file_path = Path(data_directory) / file_path
    if str(input_file_path).endswith("csv"):
        return read_collection_from_csv(input_file_path)
    else:
        pass


# def serialize_collection_as_yaml(collection: List[Message]) -> Text:
#
#
#     training_examples = OrderedDict()
#
#     # Sort by intent while keeping basic intent order
#     for example in [e.as_dict() for e in collection]:
#         if not example.get("intent"):
#             continue
#         intent = example["intent"]
#         training_examples.setdefault(intent, [])
#         training_examples[intent].append(example)
#
#     return RasaYamlWriter.


def dump_to_csv(csv_lines: List[List[Text]], file_path: Text):

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

    if str(output_file_path).endswith("csv"):
        csv_content = serialize_collection_as_csv(collection)
        dump_to_csv(csv_content, output_file_path)
    else:
        raise RuntimeError(
            "Output file path does not end with '.csv' or '.yaml'. Please provide a valid file path"
        )
