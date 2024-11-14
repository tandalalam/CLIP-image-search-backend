from typing import List, Optional, Dict

from pydantic import ValidationError

from src.controllers.qdrant_manager import QdrantManager
from flask import request, jsonify

from src.models.product import Product
from src.models.query import Query, RetrievalType


class ApiController:

    def __init__(self,
                 qdrant_configs,
                 hybrid_search_configs: Optional[Dict] = None, ):
        text_index_configs = qdrant_configs.get('text_index_configs')
        if text_index_configs:
            text_index_name = text_index_configs.get('field_name')

        self.hybrid_search_configs = hybrid_search_configs
        self.qdrant_manager = QdrantManager.get_qdrant_manager(url=qdrant_configs['db_url'],
                                                               api_key=qdrant_configs['db_api_key'],
                                                               collection_name=qdrant_configs['product_collection'],
                                                               text_index_name=text_index_name)

    def search(self):

        try:
            request_args = request.args.copy()
            query = request_args.pop('query', None)
            retrieval_type = request_args.pop('retrieval_type', RetrievalType.hybrid)
            size = request_args.pop('size', 5)
            filters = request_args.to_dict()
            query = Query(query=query,
                          retrieval_type=retrieval_type,
                          size=size,
                          filters=filters)

        except ValidationError as e:
            return jsonify({'errors': str(e.errors())}), 400

        if query.retrieval_type == RetrievalType.semantic:
            return self.__semantic_search(query)
        elif query.retrieval_type == RetrievalType.keyword:
            return self.__keyword_search(query)
        elif query.retrieval_type == RetrievalType.hybrid:
            semantic_results_percent = self.hybrid_search_configs.get('semantic_results_percent', 50)
            return self.__hybrid_search(query, semantic_results_percent)
        else:
            return jsonify({'description': 'Unsupported query type'}), 501

    def __semantic_search(self, query: Query):
        results = self.qdrant_manager.search_products_by_text(text=query.query,
                                                              top_k=query.size,
                                                              query_filter=query.filters)
        return jsonify([r['product'].to_response_obj() for r in results]), 200

    def __keyword_search(self, query: Query):
        results = self.qdrant_manager.search_products_by_keyword(text=query.query,
                                                                 top_k=query.size,
                                                                 query_filter=query.filters)
        return jsonify([r['product'].to_response_obj() for r in results]), 200

    def __hybrid_search(self, query: Query, semantic_result_percentage: int = 50):
        semantic_results = self.qdrant_manager.search_products_by_text(text=query.query,
                                                                       top_k=query.size,
                                                                       query_filter=query.filters)
        semantic_results = [res['product'] for res in semantic_results]

        keyword_results = self.qdrant_manager.search_products_by_keyword(text=query.query,
                                                                         top_k=query.size,
                                                                         query_filter=query.filters)
        keyword_results = [res['product'] for res in keyword_results]

        # combine the results
        results = semantic_results[:int(query.size * semantic_result_percentage / 100)]

        for res in keyword_results:
            if len(results) >= query.size:
                break
            if self.__not_duplicated(res, results):
                results.append(res)

        for res in semantic_results[int(query.size * semantic_result_percentage / 100):]:
            if len(results) >= query.size:
                break
            if self.__not_duplicated(res, results):
                results.append(res)

        return jsonify([r.to_response_obj() for r in results]), 200

    @staticmethod
    def __not_duplicated(product: Product, product_list: List[Product]):
        for product_item in product_list:
            if product_item.uuid == product.uuid:
                return False
        return True
