from abc import ABC, abstractmethod
from typing import List
from models import StockHolding

class BaseParser(ABC):
    name: str = "base_parser"

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """根据文件内容特征自动识别是否匹配"""
        pass

    @abstractmethod
    def parse(self, file_path: str) -> List[StockHolding]:
        """具体的解析提取逻辑"""
        pass