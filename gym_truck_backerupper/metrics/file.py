from .document import TrainingMetricsDocument
from .record import TrainingEpisodeRecord

import csv as _csv
import os as _os
import enum as _enum
import typing as _typing

class TrainingMetricsFile(TrainingMetricsDocument):
    class Header(_enum.Enum):
        EPISODE_ID  = 'episode_id'
        REWARD      = 'reward'
        LOSS        = 'loss'
        WAS_SUCCESS = 'success'
        PRECISION   = 'precision'
        STABILITY   = 'stability'
    
    _ALL_HEADERS = [Header.EPISODE_ID, Header.REWARD, Header.LOSS, Header.WAS_SUCCESS, Header.PRECISION, Header.STABILITY]
    
    def __init__(self, training_name: str, path: str) -> None:
        super().__init__(training_name)
        
        self.__path = path
        
        if _os.path.exists(path):
            self.__load()
        
        else:
            self.__create_file()
    
    def __load(self) -> None:
        with open(self.__path, 'r', encoding='utf-8', newline='') as file:
            reader = _csv.DictReader(file, delimiter=";")
            
            headers = reader.fieldnames
            
            if headers is None:
                raise ValueError("missing csv headers in provided file")
            
            if list(headers) != [header.value for header in self._ALL_HEADERS]:
                raise ValueError("CSV file headers does not correspond")

            for row in reader:
                self.add_record(
                    self._csv_row_to_episode(row)
                )
    
    def __create_file(self) -> None:
        with open(self.__path, 'w', encoding='utf-8', newline='') as file:
            writer = _csv.DictWriter(
                file,
                fieldnames=[header.value for header in self._ALL_HEADERS],
                delimiter=";"
            )

            writer.writeheader()
        
    @classmethod
    def _episode_to_csv_row(cls, episode: TrainingEpisodeRecord) -> dict[str, _typing.Any]:
        return {
            cls.Header.EPISODE_ID  .value : episode.episode_id,
            cls.Header.REWARD      .value : episode.reward,
            cls.Header.LOSS        .value : episode.loss,
            cls.Header.WAS_SUCCESS .value : episode.was_success,
            cls.Header.PRECISION   .value : episode.precision,
            cls.Header.STABILITY   .value : episode.stability
        }

    @classmethod
    def _csv_row_to_episode(cls, row: dict[str, _typing.Any]) -> TrainingEpisodeRecord:
        return TrainingEpisodeRecord(
            episode_id  = int   (row[ cls.Header.EPISODE_ID  .value ]),
            reward      = float (row[ cls.Header.REWARD      .value ]),
            loss        = float (row[ cls.Header.LOSS        .value ]),
            was_success =       (row[ cls.Header.WAS_SUCCESS .value ]).lower() in {"true", '1', 'yes'},
            precision   = float (row[ cls.Header.PRECISION   .value ]),
            stability   = float (row[ cls.Header.STABILITY   .value ])
        )
    
    def add_record(self, record: TrainingEpisodeRecord) -> None:
        super().add_record(record)
        
        with open(self.__path, 'a', encoding='utf-8', newline='') as file:
            writer = _csv.DictWriter(
                file,
                fieldnames=[header.value for header in self._ALL_HEADERS],
                delimiter=";"
            )

            writer.writerow(self._episode_to_csv_row(record))
        



