import os
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject
# 2. Import torch and immediately check if it can see the GPUs.
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
from functools import partial
import numpy as np
import jax
#jax.config.update("jax_platform_name", "cpu")
from jaxmarl import make
from jaxmarl.environments.overcooked_v2.layouts import Layout
from envs.ovc.visualizer import custom_visualizer
import jax.numpy as jnp
from functools import partial
import chex
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State, Agent, Position
custom_layout_str = """
WXWWWWWWWXW
0    R    0
1   APA   1
2    R    2
WBWWWWWWWBW
"""
import jax
import jax.numpy as jnp
from functools import partial

# --- 'get_obs_default'의 인코딩을 되돌리기 위한 헬퍼 함수 ---

@partial(jax.jit, static_argnums=(1,))
def _decode_ingredient_layers(layers, num_ingredients):
    shift = jnp.array([0, 1] + [2 * (i + 1) for i in range(num_ingredients)])
    layers_shifted = layers << shift
    ingredients_grid = jnp.sum(layers_shifted, axis=-1, dtype=jnp.int32)
    return ingredients_grid

@partial(jax.jit, static_argnums=(1,2,3))
def decode_obs_to_state(obs: chex.Array, 
                        env: OvercookedV2, 
                        n_agents: int = 2, 
                        indicate_successful_delivery: bool = False) -> State:
    
    # --- 🐞 디버깅 시작 🐞 ---
    
    H, W, C = obs.shape
    
    n = env.layout.num_ingredients

    ing_layer_size = 2 + n
    
    # ❗️❗️❗️ [오류 수정] ❗️❗️❗️
    # 주석(#)을 제거하고 실제 덧셈을 수행합니다.
    agent_layer_size = 1 + 4 + ing_layer_size

    static_layer_size = 6
    pile_layer_size = n
    extra_layer_size = 1 + (1 if indicate_successful_delivery else 0)

    # --- 1. 채널 인덱스 계산 ---
    assert n_agents == 2, "이 코드는 n_agents=2를 가정하고 작성되었습니다."
    
    idx = 0
    start_agent_0 = idx;      idx += agent_layer_size;  end_agent_0 = idx
    start_agent_1 = idx;      idx += agent_layer_size;  end_agent_1 = idx
    start_static = idx;       idx += static_layer_size; end_static = idx
    start_piles = idx;        idx += pile_layer_size;   end_piles = idx
    start_grid_ing = idx;     idx += ing_layer_size;    end_grid_ing = idx
    start_recipe = idx;       idx += ing_layer_size;    end_recipe = idx
    start_extra = idx;        idx += extra_layer_size;  end_extra = idx
    

    # --- 2. Obs 텐서 슬라이싱 ---
    obs_agent_0  = obs[..., start_agent_0:end_agent_0]
    obs_agent_1  = obs[..., start_agent_1:end_agent_1]
    obs_static   = obs[..., start_static:end_static]
    obs_piles    = obs[..., start_piles:end_piles]
    obs_grid_ing = obs[..., start_grid_ing:end_grid_ing]
    obs_recipe   = obs[..., start_recipe:end_recipe]
    obs_extra    = obs[..., start_extra:end_extra]

    # --- 3. state.grid[..., 0] (정적 객체) 복원 ---
    grid_static = jnp.zeros((H, W), dtype=jnp.int32)
    static_encoding_ids = jnp.array([
        StaticObject.WALL, StaticObject.GOAL, StaticObject.POT,
        StaticObject.RECIPE_INDICATOR, StaticObject.BUTTON_RECIPE_INDICATOR,
        StaticObject.PLATE_PILE,
    ])
    static_indices = jnp.argmax(obs_static, axis=-1)
    static_mask = jnp.any(obs_static > 0, axis=-1)
    grid_static = jnp.where(static_mask, static_encoding_ids[static_indices], grid_static)
    
    pile_encoding_ids = StaticObject.INGREDIENT_PILE_BASE + jnp.arange(n)
    pile_indices = jnp.argmax(obs_piles, axis=-1)
    pile_mask = jnp.any(obs_piles > 0, axis=-1)
    grid_static = jnp.where(pile_mask, pile_encoding_ids[pile_indices], grid_static)

    # --- 4. state.grid[..., 1] (동적 객체) 복원 ---
    grid_dynamic = _decode_ingredient_layers(obs_grid_ing, n)

    # --- 5. state.grid[..., 2] (추가 정보) 복원 ---
    grid_extra = jnp.zeros((H, W), dtype=jnp.int32)
    pot_mask = (grid_static == StaticObject.POT)
    grid_extra = jnp.where(pot_mask, obs_extra[..., 0], grid_extra)
    
    # --- 6. state.agents 복원 ---

    # 에이전트 0
    pos_0_flat = jnp.argmax(obs_agent_0[..., 0]) # Pos 채널
    pos_0_y, pos_0_x = jnp.unravel_index(pos_0_flat, (H, W))
    
    # --- 🐞 오류 발생 지점 디버깅 🐞 ---
    agent_0_slice_at_pos = obs_agent_0[pos_0_y, pos_0_x]

    dir_slice = agent_0_slice_at_pos[1:5]

    
    dir_0 = jnp.argmax(dir_slice) # Dir 채널
    inv_0_layers = obs_agent_0[pos_0_y, pos_0_x, 5:(5 + ing_layer_size)]
    inv_0 = _decode_ingredient_layers(inv_0_layers, n)

    # 에이전트 1
    pos_1_flat = jnp.argmax(obs_agent_1[..., 0]) # Pos 채널
    pos_1_y, pos_1_x = jnp.unravel_index(pos_1_flat, (H, W))
    dir_1 = jnp.argmax(obs_agent_1[pos_1_y, pos_1_x, 1:5]) # Dir 채널
    inv_1_layers = obs_agent_1[pos_1_y, pos_1_x, 5:(5 + ing_layer_size)]
    inv_1 = _decode_ingredient_layers(inv_1_layers, n)
    
    agents = Agent(
        pos=Position(x=jnp.array([pos_0_x, pos_1_x]), y=jnp.array([pos_0_y, pos_1_y])),
        dir=jnp.array([dir_0, dir_1], dtype=jnp.int32),
        inventory=jnp.array([inv_0, inv_1], dtype=jnp.int32)
    )

    # --- 7. state.recipe 복원 ---
    recipe_indicator_mask = (grid_static == StaticObject.RECIPE_INDICATOR) | \
                            (grid_static == StaticObject.BUTTON_RECIPE_INDICATOR)
    
    first_recipe_tile_idx = jnp.argmax(recipe_indicator_mask.flatten())
    recipe_y, recipe_x = jnp.unravel_index(first_recipe_tile_idx, (H, W))
    
    recipe_layers = obs_recipe[recipe_y, recipe_x]
    recipe = _decode_ingredient_layers(recipe_layers, n)

    # --- 8. State 객체 조립 ---
    grid = jnp.stack([grid_static, grid_dynamic, grid_extra], axis=-1)
    
    restored_state = State(
        agents=agents, grid=grid, recipe=recipe,
        time=jnp.array(0), terminal=jnp.array(False),
        new_correct_delivery=jnp.array(False),
        ingredient_permutations=None
    )

    return restored_state


l =Layout.from_string(custom_layout_str)
env = make("overcooked_v2", layout=l, agent_view_size = None,random_agent_positions=True,random_reset = False,max_steps=400,sample_recipe_on_delivery=True)
key = jax.random.PRNGKey(0)
vizu = custom_visualizer()
key, reset_key = jax.random.split(key)
initial_obs, initial_state = env.reset(reset_key)
s = vizu._render_state(initial_state,None)
g = env.get_obs_default(initial_state)
rs = decode_obs_to_state(g[0],env)
rs = vizu._render_state(rs,None)
s = np.array(s)
o = np.array(rs)
import matplotlib.pyplot as plt
plt.imsave('state.png',s)
plt.imsave('obs.png',o)