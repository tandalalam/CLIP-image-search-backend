import json

from loguru import logger

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


def create_full_text_index():
    config_manager = ConfigManager.get_config_manager()
    db_configs = config_manager.get_prop('qdrant_configs')

    qdrant_manager = QdrantManager.get_qdrant_manager(url=db_configs['db_url'],
                                                      api_key=db_configs['db_api_key'],
                                                      collection_name=db_configs['product_collection'])

    keyword_index_configs = db_configs.get('text_index_configs')
    qdrant_manager.index_keywords(field_name=keyword_index_configs.pop('field_name'),
                                  params=keyword_index_configs)


def start_application():
    config_manager = ConfigManager.get_config_manager()
    qdrant_config = config_manager.get_prop('qdrant_configs')
    server_config = config_manager.get_prop('server_configs')
    hybrid_search_configs = config_manager.get_prop('hybrid_search_configs')

    app = Flask(__name__)
    api_controller = ApiController(qdrant_configs=qdrant_config,
                                   hybrid_search_configs=hybrid_search_configs)

    app.add_url_rule('/search',
                     'semantic_search',
                     view_func=api_controller.search,
                     methods=['GET'])

    app.add_url_rule('/index',
                     'index',
                     view_func=api_controller.index,
                     methods=['GET'])

    app.add_url_rule('/is_ready',
                     'index',
                     view_func=api_controller.is_ready,
                     methods=['GET'])

    return app, server_config


if __name__ == '__main__':
    app, server_configs = start_application()

    port = ConfigManager.get_config_manager().get_prop('service_configs').get('port')
    from waitress import serve

    logger.info('Starting server...')

    serve(app, host='0.0.0.0',
          port=port,
          threads=5)
