from typing import List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions
from utils.clip_encoder import CLIPEncoder
from models.product import Product
from loguru import logger


class QdrantManager:
    qdrant_manager = None

    @staticmethod
    def get_qdrant_manager(url, api_key, collection_name, text_index_name=None):
        if QdrantManager.qdrant_manager is None:
            QdrantManager.qdrant_manager = QdrantManager(url, api_key, collection_name, text_index_name)
        return QdrantManager.qdrant_manager

    def __init__(self, url: str, api_key: str, collection_name: str, text_index_name: str | None = None):
        self.client = QdrantClient(url=url,
                                   api_key=api_key)
        self.collection_name = collection_name
        self.clip_encoder = CLIPEncoder()
        self.text_index_name = text_index_name
        # Create collection if it doesn't exist
        self.__ensure_collection()

    def __ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except exceptions.UnexpectedResponse as e:
            if e.status_code == 404:  # Not Found
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.clip_encoder.model.config.projection_dim,
                        distance=models.Distance.COSINE,
                    )
                )

    def index_keywords(self, field_name: str, params: Dict):
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.TextIndexParams(
                    type="text",
                    **params
                ),
            )
            logger.info('Successfully indexed keywords.')

        except exceptions.UnexpectedResponse as e:
            logger.error(f'Error occurred, indexing {field_name}.')
            logger.error(e)

    def insert_batch(self, products: List[Product], insertion_batch_size=64):
        for batch_start in range(1410, len(products), insertion_batch_size):
            batch_end = batch_start + insertion_batch_size

            batch_points = []

            logger.info(f'start encoding items [{batch_start}, {batch_end}])')

            for product in products[batch_start:batch_end]:
                try:
                    product_encoding = self.clip_encoder.encode_image(images=[image for image in product.images],
                                                                      is_url=True)
                    vector_record = product.to_vector_record(product_encoding)
                    batch_points.append(models.PointStruct(**vector_record))
                except ValueError as e:
                    logger.error(f'error encoding images {[image for image in product.images]}')
                    logger.error(e)

            logger.info(f'start inserting items [{batch_start}, {batch_end}])')

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )

    def add_product(self, product: Product):
        image_embedding = self.clip_encoder.encode_image(product.image_url, is_url=True)
        vector_record = product.to_vector_record(image_embedding)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(**vector_record)],
        )

    def search_products_by_text(self,
                                text: str,
                                top_k: int = 10,
                                query_filter: models.Filter | None = None) -> List[Dict]:

        text_embedding = self.clip_encoder.encode_text(text)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_filter=query_filter,
            query_vector=text_embedding,
            limit=top_k,
        )

        results = []
        for point in search_result:
            product = Product(**point.payload)
            results.append({'product': product, 'similarity_score': point.score})

        return results

    def search_products_by_keyword(self,
                                   text: str,
                                   top_k: int = 10,
                                   query_filter: models.Filter | None = None) -> List[Dict]:

        text_filter = models.FieldCondition(
            key=self.text_index_name,
            match=models.MatchText(text=text),
        )

        if query_filter is not None:
            query_filter.must.append(text_filter)
        else:
            query_filter = models.Filter(
                must=[text_filter],
            )

        search_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in search_result[0]:
            product = Product(**point.payload)
            results.append({'product': product})

        return results
