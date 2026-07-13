from pathlib import Path
import os
import hydra
# os.environ['WANDB_BASE_URL']='https://api.bandw.top'
os.environ['WANDB_BASE_URL']='https://api.wandb.ai'

@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.workspace import Workspace

    root_dir = Path.cwd()

    workspace = Workspace(cfg)

    snapshot = workspace.work_dir/'snapshots/snapshot.pt'
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
