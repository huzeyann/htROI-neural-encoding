try:
    from clearml import Task

    clearml_installed = True
except:
    clearml_installed = False

from src.train import train_main
from src.config import get_cfg_defaults

if __name__ == '__main__':

    cfg = get_cfg_defaults()

    if clearml_installed:
        task = Task.init(
            project_name='Algonauts2021',
            task_name='task template',
            tags=None,
            reuse_last_task_id=False,
            continue_last_task=False,
            output_uri=None,
            auto_connect_arg_parser=True,
            auto_connect_frameworks=True,
            auto_resource_monitoring=True,
            auto_connect_streams=True,
        )
        task.connect(cfg)
        task_id = task.id
    else:
        task_id = 'debug'

    train_main(cfg, task_id)
