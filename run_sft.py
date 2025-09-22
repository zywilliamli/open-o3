import sky
import textwrap
from dotenv import dotenv_values, load_dotenv
from sky import ClusterStatus
import os

load_dotenv()


def launch_model():
    setup_script = textwrap.dedent(
        """
            echo 'Setting up environment...'
            apt update && apt install -y nvtop curl unzip
            
            # Install AWS CLI v2 (Python 3 compatible)
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip awscliv2.zip
            ./aws/install
            
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.local/bin/env
        """
    )

    run_script = textwrap.dedent(f"""
        uv add unsloth
        DEBUG=1 ADD_SPECIAL_TOKENS=1 uv run python sft.py
    """)

    # Create a SkyPilot Task
    task = sky.Task(
        name=f"open-o3-sft",
        setup=setup_script,
        run=run_script,
        workdir=".",  # Sync the project directory
        envs=dict(dotenv_values()),  # type: ignore
    )
    task.set_resources(sky.Resources(accelerators="H200-SXM:1"))

    # Generate cluster name
    cluster_name = f"open-o3-sft"
    # Add cluster prefix if defined in environment
    cluster_prefix = os.environ.get("CLUSTER_PREFIX")
    if cluster_prefix:
        cluster_name = f"{cluster_prefix}-{cluster_name}"
    print(f"Launching task on cluster: {cluster_name}")

    print("Checking for existing cluster and jobs…")
    cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
    if len(cluster_status) > 0 and cluster_status[0]["status"] == ClusterStatus.UP:
        print(f"Cluster {cluster_name} is UP. Canceling any active jobs…")
        sky.stream_and_get(sky.cancel(cluster_name, all=True))

    # Launch the task; stream_and_get blocks until the task starts running, but
    # running this in its own thread means all models run in parallel.
    job_id, _ = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=60,
            down=True,
        )
    )

    print(f"Job submitted(ID: {job_id}). Streaming logs…")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} finished with exit code {exit_code}.")


launch_model()
