from pydantic import BaseModel, Field
from typing_extensions import Annotated, Optional


class OptimizationConfig(BaseModel):
    weight_decay: Annotated[
        Optional[float],
        Field(
            description="The weight decay parameter to use for optimization",
            default=1.0e-4,
        ),
    ]
    learning_rate: Annotated[
        Optional[float],
        Field(
            description="The learning rate parameter to use for optimization",
            default=1.0e-4,
        ),
    ]


class TrainingConfig(BaseModel):
    model_name: Annotated[
        str, Field(description="Name of the model to load from timm.")
    ]
    pretrained: Annotated[
        Optional[bool], Field(description="Use a pretrained model as base", default=True)
    ]
    num_epochs: Annotated[
        Optional[int], Field(description="Number of epochs to run", default=50)
    ]
    batch_size: Annotated[
        Optional[int], Field(description="The batch size to use.", default=128)
    ]
    optimization: Annotated[
        OptimizationConfig,
        Field(
            description="Optimization configuration", default_factory=OptimizationConfig
        ),
    ]
    n_classes: Annotated[
        Optional[int], Field(description="Number of classes in the data", default=8)
    ]
    n_workers: Annotated[
        Optional[int],
        Field(description="Number of workers to use when loading data", default=4),
    ]
