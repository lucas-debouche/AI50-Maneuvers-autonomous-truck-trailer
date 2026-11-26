import dataclasses as _dataclasses

@_dataclasses.dataclass(frozen=True)
class TrainingEpisodeRecord:
    episode_id: int
    
    reward: float
    loss: float
    
    was_success: bool
    
    precision: float
    stability: float


