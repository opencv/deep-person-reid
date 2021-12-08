"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

def run_acc_aware_training_loop(engine, nncf_config, configure_optimizers_fn):
    from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop

    # Use only main model
    model = engine.models[engine.main_model_name]
    optimizer = engine.optims[engine.main_model_name]
    scheduler = engine.scheds[engine.main_model_name]

    def _configure_optimizers_fn():
        return optimizer, scheduler

    def dec_test(func):
        def inner(model, epoch):
            top1, _ = func(epoch)
            return top1
        return inner

    def dec_train(func):
        def inner(*args, **kwargs):
            engine.epoch = kwargs['epoch']
            func()
            return

        return inner

    engine.train_data_loader = engine.train_loader
    engine.max_epoch = nncf_config.get('accuracy_aware_training').get('params').get('maximal_total_epochs', 200)

    validate_fn = dec_test(engine.test)
    train_fn = dec_train(engine.train)

    acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, engine.compression_ctrl,
                                                                  lr_updates_needed=False)
    model = acc_aware_training_loop.run(model,
                                        train_epoch_fn=train_fn,
                                        validate_fn=validate_fn,
                                        configure_optimizers_fn=_configure_optimizers_fn)
    return model
