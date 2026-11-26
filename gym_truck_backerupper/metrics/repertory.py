from .document import TrainingMetricsDocument

import typing as _typing

_D = _typing.TypeVar("_D", bound=TrainingMetricsDocument)

class TrainingMetricsRepertory(_typing.Generic[_D]):
    def __init__(self) -> None:
        self.__documents_cache: dict[str, _D] = {}
        
    def register_document(self, document: _D) -> None:
        self.__documents_cache[document.training_name] = document
    
    def get_document(self, training_name: str) -> _D:
        if not training_name in self.__documents_cache:
            raise ReferenceError(f"unknown document {training_name!r}")
        
        return self.__documents_cache[training_name]
    
    def knows_document(self, training_name: str) -> bool:
        return training_name in self.__documents_cache
    
