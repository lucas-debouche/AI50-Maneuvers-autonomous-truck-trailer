from .repertory import TrainingMetricsRepertory
from .file import TrainingMetricsFile

import os as _os
import sys as _sys

class TrainingMetricsDirectory(TrainingMetricsRepertory[TrainingMetricsFile]):
    def __init__(self, path: str) -> None:
        super().__init__()
        
        self.__path = path
        
        if _os.path.exists(path):
            self.__load_documents()
        else:
            _os.makedirs(path)
    
    def __load_documents(self) -> None:
        for filename in _os.listdir(self.__path):
            training_name, file_ext = (filename.rsplit('.', 1) + [''])[:2]
            
            if file_ext.lower() == 'csv':
                try:
                    document = TrainingMetricsFile(training_name, self.__path + _os.path.sep + filename)
                    self.register_document(document)
                
                except Exception as exc:
                    print(f"Could not read document {filename} in directory {self.__path} :", str(exc), file=_sys.stderr)
    
    def get_document(self, training_name: str) -> TrainingMetricsFile:
        if not self.knows_document(training_name):
            try:
                document = TrainingMetricsFile(training_name, self.__path + _os.path.sep + training_name + '.csv')
                self.register_document(document)
            
            except Exception as exc:
                print(f"Could not create document {training_name + 'csv'} in directory {self.__path} :", str(exc), file=_sys.stderr)
        
        return super().get_document(training_name)

