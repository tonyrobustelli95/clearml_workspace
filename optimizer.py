from clearml import Task, TaskTypes
from clearml.automation import *

if __name__ == "__main__":

    task = Task.init(project_name='clearml-init', task_name='Autoencoder optimization', task_type=TaskTypes.optimizer)

    optimizer = HyperParameterOptimizer(base_task_id="67b176a458844d44888a92df6a0652d8", 
                                        hyper_parameters=[UniformIntegerParameterRange('filters', 5, 5)],
    objective_metric_title='accuracy',
    objective_metric_series='validation',
    objective_metric_sign='min',
    )

    """
    Extra parameters for HyperParameterOptimizer (HPO)
    UniformIntegerParameterRange('batch_size', 128, 256, step_size=64)
    max_number_of_concurrent_tasks=5 
    """

    optimizer.start()

    optimizer.wait()

    optimizer.stop()