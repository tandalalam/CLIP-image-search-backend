from pydantic import ValidationError

from controllers.qdrant_manager import QdrantManager
from flask import request, jsonify
from models.query import Query, RetrievalType
from loguru import logger


class ApiController:

    def __init__(self,
                 qdrant_configs):
        self.qdrant_manager = QdrantManager.get_qdrant_manager(url=qdrant_configs['db_url'],
                                                               api_key=qdrant_configs['db_api_key'],
                                                               collection_name=qdrant_configs['product_collection'])

    def semantic_search(self):
        try:
            request_args = request.args.copy()
            query = request_args.pop('query', None)
            retrieval_type = request_args.pop('retrieval_type', RetrievalType.hybrid)
            size = request_args.pop('size', 5)
            filters = request_args.to_dict()
            logger.info(filters)
            query = Query(query=query,
                          retrieval_type=retrieval_type,
                          size=size,
                          filters=filters)
        except ValidationError as e:
            return jsonify({'errors': str(e.errors())}), 400

        if query.retrieval_type == RetrievalType.hybrid:
            results = self.qdrant_manager.search_products_by_text(text=query.query,
                                                                  top_k=query.size,
                                                                  query_filter=query.filters)

            response = jsonify([r['product'].to_response_obj() for r in results]), 200
        else:
            response = jsonify({'description': 'Unsupported query type'}), 501
        return response
