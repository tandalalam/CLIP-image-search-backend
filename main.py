import json
from controllers.api_controller import ApiController
from controllers.qdrant_manager import QdrantManager
from configs.configs import ConfigManager
from utils.products_preprocessor import ProductsPreprocessor
from flask import Flask


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


def start_application():
    config_manager = ConfigManager.get_config_manager()
    qdrant_config = config_manager.get_prop('qdrant_configs')
    server_config = config_manager.get_prop('server_configs')

    app = Flask(__name__)
    api_controller = ApiController(qdrant_configs=qdrant_config)

    app.add_url_rule('/search',
                     'semantic_search',
                     view_func=api_controller.semantic_search,
                     methods=['GET'])

    return app, server_config


if __name__ == '__main__':
    app, server_configs = start_application()
    app.run(debug=True,
            host='0.0.0.0',
            port=server_configs.get('port'))
