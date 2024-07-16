from clearml import Task, TaskTypes
from clearml.automation import *
import argparse

if __name__ == "__main__":

    task = Task.init(project_name='clearml-init', task_name='Autoencoder optimization', task_type=TaskTypes.optimizer)
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Script to do optimization using a task ID")
    parser.add_argument('--task_id', type=str, help='task ID input')

    args = parser.parse_args()
    task_id = str(args.task_id)

    task.execute_remotely(queue_name="default")

    optimizer = HyperParameterOptimizer(base_task_id=task_id, 
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