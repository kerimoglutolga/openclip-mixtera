
import os 

from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import CC12MDataset

from mixtera.core.datacollection.index.parser.parser_collection import GenericMetadataParser
    
def parsing_func(sample):
    return sample

if __name__ == "__main__":
    host = os.environ.get("MIXTERA_SERVER_ADDR")
    port = int(os.environ.get("MIXTERA_SERVER_PORT"))

    dataset_name = os.environ.get("DATASET_NAME")
    dataset_path = os.environ.get("DATASET_PATH")

    print(f"Registering dataset {dataset_name} at {dataset_path} to Mixtera server at {host}:{port}")

    client = MixteraClient.from_remote(host=host, port=port)
    client.register_metadata_parser("GenericMetadataParser", GenericMetadataParser)
    client.register_dataset(dataset_name, dataset_path, CC12MDataset, parsing_func, "GenericMetadataParser")
        