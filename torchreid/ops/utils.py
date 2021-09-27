'''
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
 '''

class EvalModeSetter:
    def __init__(self, module, m_type):
        self.modules = module
        if not isinstance(self.modules, (tuple, list)):
            self.modules = [self.modules]

        self.modes_storage = [{} for _ in range(len(self.modules))]

        self.m_types = m_type
        if not isinstance(self.m_types, (tuple, list)):
            self.m_types = [self.m_types]

    def __enter__(self):
        for module_id, module in enumerate(self.modules):
            modes_storage = self.modes_storage[module_id]

            for child_name, child_module in module.named_modules():
                matched = any(isinstance(child_module, m_type) for m_type in self.m_types)
                if matched:
                    modes_storage[child_name] = child_module.training
                    child_module.train(mode=False)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module_id, module in enumerate(self.modules):
            modes_storage = self.modes_storage[module_id]

            for child_name, child_module in module.named_modules():
                if child_name in modes_storage:
                    child_module.train(mode=modes_storage[child_name])
