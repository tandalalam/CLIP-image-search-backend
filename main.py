import json
from controllers.qdrant_manager import QdrantManager
from configs.configs import ConfigManager
from utils.products_preprocessor import ProductsPreprocessor


def index_products():
    config_manager = ConfigManager.get_config_manager()
    db_configs = config_manager.get_prop('qdrant_configs')
    job_configs = config_manager.get_prop('insertion_job_configs')

    qdrant_manager = QdrantManager.get_qdrant_manager(url=db_configs['db_url'],
                                                      api_key=db_configs['db_api_key'],
                                                      collection_name=db_configs['product_collection'])

    # load products json
    data_path = job_configs['path_to_products']
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)

    product_preprocessor = ProductsPreprocessor()
    product_preprocessor.process_products(data)

    qdrant_manager.insert_batch(products=product_preprocessor.products,
                                insertion_batch_size=job_configs['insertion_batch_size'], )


if __name__ == '__main__':
    index_products()
