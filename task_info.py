from clearml import Task, TaskTypes
from clearml.automation import *
import argparse

if __name__ == "__main__":

    task = Task.init(project_name='clearml-init', task_name='Autoencoder info')
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Script to do optimization using a task ID")
    parser.add_argument('--task_id', type=str, help='task ID input')

    args = parser.parse_args()
    task_id = str(args.task_id)

    task.execute_remotely(queue_name="default")

    task_retrived = Task.get_task(task_id=task_id)

    print(task_retrived)