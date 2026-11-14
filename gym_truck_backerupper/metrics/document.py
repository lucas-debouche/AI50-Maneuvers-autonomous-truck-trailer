from .record import TrainingEpisodeRecord

import typing as _typing

class TrainingMetricsDocument:
    def __init__(self, training_name: str) -> None:
        self.__training_name = training_name
        self.__records: list[TrainingEpisodeRecord] = []
        
    @property
    def training_name(self) -> str:
        return self.__training_name
    
    @property
    def all_records(self) -> _typing.Sequence[TrainingEpisodeRecord]:
        return self.__records

    def add_record(self, record: TrainingEpisodeRecord):
        self.__records.append(record)
    


