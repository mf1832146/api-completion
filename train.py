import sys

from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch import optim
import dataset
import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, TopKCategoricalAccuracy
from ignite.contrib.handlers.tensorboard_logger import *


class Solver:
    def __init__(self, args, model, api_dict, class_dict, class_to_api_dict):
        self.args = args
        self.model = model
        self.api_dict = api_dict
        self.class_dict = class_dict
        self.class_to_api_dict = class_to_api_dict

    def train(self):
        train_loader, valid_loader = dataset.get_data_loaders(self.api_dict,
                                                              self.class_dict,
                                                              self.class_to_api_dict,
                                                              self.args)
        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            print('use gpu')

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = create_supervised_trainer(self.model, optimizer, criterion, device)

        if sys.version_info > (3,):
            from ignite.contrib.metrics.gpu_info import GpuInfo

            try:
                GpuInfo().attach(trainer)
            except RuntimeError:
                print(
                    "INFO: By default, in this example it is possible to log GPU information (used memory, utilization). "
                    "As there is no pynvml python package installed, GPU information won't be logged. Otherwise, please "
                    "install it : `pip install pynvml`"
                )

        metrics = {"top-1 acc": TopKCategoricalAccuracy(k=1), "loss": Loss(criterion)}
        train_evaluator = create_supervised_evaluator(self.model, metrics, device)
        validation_evaluator = create_supervised_evaluator(self.model, metrics, device)

        # save model
        save_handler = ModelCheckpoint('train/models/'+self.args.model, n_saved=10,
                                       filename_prefix='prefix',
                                       create_dir=True)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), save_handler, {self.args.model: self.model})

        # early stop
        early_stop_handler = EarlyStopping(patience=10, score_function=self.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            train_evaluator.run(train_loader)
            validation_evaluator.run(valid_loader)

        tb_logger = TensorboardLogger(self.args.log_dir + self.args.model + '/')

        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(
                tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all"
            ),
            event_name=Events.ITERATION_COMPLETED(every=50),
        )

        tb_logger.attach(
            train_evaluator,
            log_handler=OutputHandler(
                tag="training", metric_names=["loss"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        tb_logger.attach(
            validation_evaluator,
            log_handler=OutputHandler(tag="validation", metric_names=["loss", "top-1 acc"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        trainer.run(train_loader, max_epochs=self.args.max_epoch)
        tb_logger.close()

    @staticmethod
    def score_function(engine):
        top_1_acc = engine.state.metrics['top-1 acc']
        return top_1_acc






