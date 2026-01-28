import dataclasses
import functools
import logging
import platform
import os
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import sys

import orbax.checkpoint as ocp

# [DIAGRAM: Training Flow]
# Stage 1: Init Params -> Mask Action Stream (Freeze) -> Train Tactile Stream -> Save Checkpoint
# Stage 2: Load Checkpoint (Params Only) -> Reset Optimizer -> Unfreeze All -> Joint Train


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        # Fallback if no handler exists
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    
    # --- CHANGE START ---
    # Relax strict checking to allow partial loading (e.g., when adding new layers like torque_proj)
    try:
        at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    except ValueError as e:
        logging.warning(f"Detailed weight structure validation failed (proceeding with partial load): {e}")
    # --- CHANGE END ---

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )



def create_freeze_filter(config: _config.TrainConfig):
    """
    Creates a filter to freeze the Action Stream components.
    """
    # Config에 freeze_action_stream 속성이 없으면 False 처리
    if not getattr(config, "freeze_action_stream", False):
        return config.trainable_filter

    logging.info("\033[93m[Freeze Config] Freezing Action Stream active! Gradients will be blocked for action components.\033[0m")

    # Regex to match Action Stream components (Pi0 Architecture specific)
    # action_in_proj: Action 입력 projection
    # action_out_proj: Action 출력 projection
    # time_mlp / action_time_mlp: Action stream의 time conditioning
    # layers/.*_2: Gemma의 Action Expert (Index 2)
    action_stream_regex = r".*(action_in_proj|action_out_proj|time_mlp|state_proj|action_time_mlp|layers/.*_2|final_norm_2).*"

    def filter_fn(path, param):
        # 1. 먼저 기본 Trainable 필터 통과 여부 확인
        is_trainable_type = config.trainable_filter(path, param)
        if not is_trainable_type:
            return False
        
        # 2. Action Stream 패턴과 매칭되면 False (Freeze)
        if nnx_utils.PathRegex(action_stream_regex)(path, param):
            return False 
        
        return True # Tactile Stream 및 Shared Component는 학습

    return filter_fn

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, 
    init_rng: at.KeyArrayLike, 
    mesh: jax.sharding.Mesh, 
    *, 
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    
    active_trainable_filter = create_freeze_filter(config)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        # Stage 2: Load checkpoint params
        if partial_params is not None:
            logging.info("Loading params from Stage 1 checkpoint...")
            graphdef, state = nnx.split(model)
            
            # Convert partial_params back to NNX State if it's a pure dict
            if isinstance(partial_params, dict):
                state = state.replace_by_pure_dict(partial_params)
            else:
                state = partial_params
                
            model = nnx.merge(graphdef, state)
            logging.info("✓ Successfully loaded Stage 1 parameters")

        params = nnx.state(model)
        
        # Apply dtype conversion to frozen params
        params = nnx_utils.state_map(
            params, 
            config.freeze_filter, 
            lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )

        # Initialize optimizer with current trainable filter
        opt_state = tx.init(params.filter(active_trainable_filter))

        return training_utils.TrainState(
            step=0,  # Always start from 0 for Stage 2
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=opt_state,
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # --- Weight Loading Logic ---
    partial_params = None
    
    # Case A: Load from Stage 1 checkpoint (Stage 2)
    if config.checkpoint_path:
        logging.info(f"[Stage 2] Loading from: {config.checkpoint_path}")
        
        ckpt_path_obj = epath.Path(config.checkpoint_path)
        
        if not ckpt_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path_obj}")
        
        try:
            # Create checkpointer
            checkpointer = ocp.PyTreeCheckpointer()
            
            # Method 1: Try loading full TrainState then extract params
            logging.info("Attempting to load full TrainState...")
            try:
                full_state = checkpointer.restore(
                    ckpt_path_obj,
                    item=train_state_shape
                )
                
                # Extract only params
                partial_params = full_state.params
                if hasattr(partial_params, 'to_pure_dict'):
                    partial_params = partial_params.to_pure_dict()
                    
                logging.info("✓ Loaded params from full TrainState")
                
            except Exception as e1:
                logging.warning(f"Could not load full state: {e1}")
                
                # Method 2: Try loading params subdirectory
                params_path = ckpt_path_obj / 'params'
                if params_path.exists():
                    logging.info(f"Trying params subdirectory: {params_path}")
                    
                    # Get target structure
                    params_shape = train_state_shape.params
                    if hasattr(params_shape, 'to_pure_dict'):
                        params_shape = params_shape.to_pure_dict()
                    
                    partial_params = checkpointer.restore(
                        params_path,
                        item=params_shape
                    )
                    logging.info("✓ Loaded params from subdirectory")
                else:
                    raise ValueError(f"Could not load checkpoint from {ckpt_path_obj}")
                    
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.error(f"Checkpoint structure at {ckpt_path_obj}:")
            for item in ckpt_path_obj.iterdir():
                logging.error(f"  - {item.name}")
            raise
    
    # Case B: Generic weight loading
    elif config.weight_loader:
        logging.info("Loading from weight_loader...")
        params_shape = train_state_shape.params
        if hasattr(params_shape, 'to_pure_dict'):
            params_shape = params_shape.to_pure_dict()
        partial_params = _load_weights_and_validate(config.weight_loader, params_shape)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    
    # Determine which filter to use for gradients
    active_trainable_filter = create_freeze_filter(config)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions,
        tactile_future: at.Float[at.Array, "..."] | None = None,
        torque_future: at.Float[at.Array, "..."] | None = None
    ):
        # Prepare kwargs for compute_loss based on what's available
        extra_args = {}
        if tactile_future is not None:
            extra_args["tactile_future"] = tactile_future
        if torque_future is not None:
            extra_args["torque_future"] = torque_future
        
        #import pdb; pdb.set_trace()
        chunked_loss = model.compute_loss(rng, observation, actions, train=True, **extra_args)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, tactile_history, tactile_future, torque_history, torque_future = batch

    # Need to inject tactile_history into observation if model expects it there
    if tactile_history is not None:
        observation = dataclasses.replace(observation, tactile_history=tactile_history)
    if torque_history is not None:
        observation = dataclasses.replace(observation, torque_history=torque_history)

    # Filter out frozen params from gradient calculation
    diff_state = nnx.DiffState(0, active_trainable_filter)

    extra_args = {}
    if tactile_future is not None:
        extra_args["tactile_future"] = tactile_future
    if torque_future is not None:
        extra_args["torque_future"] = torque_future
    
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions, **extra_args
    )

    params = state.params.filter(active_trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # --- CLI Args Handling (Simulation) ---
    # Since config is frozen/parsed before, we check sys.argv or environment for custom flags
    # In a real setup, these should be in _config.py definition.
    # We monkey-patch/inject for this script.
    

    if "--freeze_action_stream" in sys.argv:
        logging.info("\033[93m[CLI] Detected --freeze_action_stream flag. Injecting into config.\033[0m")
        # config가 Frozen Dataclass일 경우를 대비해 우회하거나 replace 사용
        try:
            config = dataclasses.replace(config, freeze_action_stream=True)
        except TypeError: 
            # dataclass 정의에 필드가 없는 경우 runtime monkey patch
            # 주의: dataclass가 frozen=True이면 setattr 실패할 수 있음. 이 경우 _config.py 수정 필요.
            # 여기서는 임시방편으로 setattr 시도
            try:
                object.__setattr__(config, "freeze_action_stream", True)
            except Exception as e:
                logging.error(f"Failed to set freeze_action_stream on config: {e}. Please add 'freeze_action_stream: bool = False' to your TrainConfig.")
                # Fallback: create_freeze_filter에서 sys.argv를 직접 체크하는 것이 안전할 수 있으나, 여기선 주입되었다고 가정.

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by devices {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    
    # Init Train State (Params Load + Opt Reset logic inside)
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    print("RESUMING: ", resuming)
    if resuming:
        logging.info("Resuming full training state (optimizer + step)...")
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    logging.info(f"Starting training loop from step {start_step} to {config.num_train_steps}")
    
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
 
    infos = []
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            wandb.log(reduced_info, step=step)
            infos = []
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    checkpoint_manager.wait_until_finished()

if __name__ == "__main__":
    main(_config.cli())