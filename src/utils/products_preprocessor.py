from typing import List, Dict, Optional
from pydantic_core._pydantic_core import ValidationError
from models.product import Product
from loguru import logger


class ProductsPreprocessor:

    def __init__(self, products_list: Optional[List[Dict]] = None):
        self.products_list = None
        self.products = None
        if products_list is not None:
            self.process_products(products_list)

    def process_products(self, products_list: List[Dict]):
        self.products_list = products_list
        self.__initialize_products()

    def __initialize_products(self):
        def create_product(product: Dict):
            try:
                return Product(**product)
            except ValidationError as e:
                logger.error('Validation Error while creating product {product}'.format(product=product))

        self.products = [create_product(product) for product in self.products_list]
        self.products = [product for product in self.products if product is not None]
