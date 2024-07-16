from clearml import PipelineController, Task

pipeline_task = Task.init(project_name='clearml-pipeline', task_name='Pipeline remote execution')

queue_name = "default"

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
    execution_queue=queue_name
)

# Finds the task_id related to the previous step/task by navigating the pipeline's DAG
id_step1 = pipeline.get_pipeline_dag()['autoencoder_step'].base_task_id

# Add the second task to the pipeline by also passing the task_id of the first task
pipeline.add_step(
    name='optimizer_step',
    base_task_project='clearml-init',
    base_task_name="Autoencoder optimization",
    parents=['autoencoder_step'],
    parameter_override={
        'Args/task_id': id_step1
    },
    execution_queue=queue_name
)

# Runs the pipeline
pipeline.start(queue=queue_name)