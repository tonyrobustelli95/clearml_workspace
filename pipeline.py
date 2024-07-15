from clearml import Task, PipelineController

# Defines the Pipeline's controller
pipeline = PipelineController(
    name='Pipeline Example',
    project="clearml-pipeline",
    version='0.1'
)

# Add the first task to the pipeline
pipeline.add_step(
    name='autoencoder_step',
    base_task_project='clearml-init',
    base_task_name="Autoencoder training",
    execution_queue="default"
)

# Add the second task to the pipeline by also passing the task_id of the first task
pipeline.add_step(
    name='optimizer_step',
    base_task_project='clearml-init',
    base_task_name="Autoencoder optimization",
    parents=['autoencoder_step'],
    parameter_override={
        'Args/task_id': '$autoencoder_step.task_id'
    },
    execution_queue="default"
)

# Runs the pipeline
pipeline.start(queue="default")