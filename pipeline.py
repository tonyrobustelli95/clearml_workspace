from clearml import Task, PipelineController

def create_task(project_name, task_name, script, taskType):
    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        script=script,
        task_type=taskType
    )
    Task.enqueue(task=task,queue_name='default')
    return task

task1 = create_task('clearml-init', "Autoencoder training", 'autoencoder.py', Task.TaskTypes.training)
task2 = create_task('clearml-init', "Autoencoder optimization", 'optimizer.py', Task.TaskTypes.optimizer)

# Defines the Pipeline's controller
pipeline = PipelineController(
    name='Pipeline Example',
    project="clearml-pipeline",
    version='0.1'
)

# Add the first task to the pipeline
pipeline.add_step(
    name='autoencoder_step',
    base_task_id=task1.id,
    execution_queue="default"
)

# Add the second task to the pipeline by also passing the task_id of the first task
pipeline.add_step(
    name='optimizer_step',
    base_task_id=task2.id,
    parents=['autoencoder_step'],
    parameter_override={
        'Args/task_id': task1.id
    },
     execution_queue="default"
)

# Runs the pipeline
pipeline.start(queue='default')