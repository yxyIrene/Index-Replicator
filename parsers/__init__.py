from typing import List
from .base import BaseParser
from .parser_format1 import Format1Parser
from .parser_format2 import Format2Parser
from .parser_liman import LimanParser
from .parser_pb_summary import PBSummaryParser
from .parser_hti_inventory import HTIInventoryParser
from .parser_exclusive import ExclusiveMasterParser
from .parser_intraday_inventory import HTIIntradayInventoryParser

EXCLUSIVE_EOD_PARSER = ExclusiveMasterParser()
INTRADAY_PARSER = HTIIntradayInventoryParser()

STANDARD_PARSERS = [
    HTIInventoryParser(),
    PBSummaryParser(),
    LimanParser(),
    Format1Parser(),
    Format2Parser(),
]

def get_parser(file_path: str, run_mode: str = "standard", exclude_sources: List[str] = None) -> BaseParser:
    if run_mode == "intraday":
        if INTRADAY_PARSER.can_handle(file_path):
            INTRADAY_PARSER.exclude_sources = [s.upper().strip() for s in (exclude_sources or [])]
            return INTRADAY_PARSER
        return None

    if run_mode == "exclusive_eod":
        if EXCLUSIVE_EOD_PARSER.can_handle(file_path):
            return EXCLUSIVE_EOD_PARSER
        return None

    for parser in STANDARD_PARSERS:
        if parser.can_handle(file_path):
            return parser
    return None