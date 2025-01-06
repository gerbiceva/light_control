from utils import FrameLimiter
# from graph import Graph

# def start() -> list[int]:
#     return [1]

# def plus_one(a: list[int]) -> list[int]:
#     return [x+1 for x in a]

# def out(a: list[int], b: list[int]):
#     print(a, b)

# G = Graph([start, plus_one, plus_one, out],
#             {
#             (0, 0): [(1,0), (2,0)],
#             (1, 0): [(3,0)],
#             (2, 0): [(3,1)],
#             })
# G.evaluate()
# G.evaluate()

from nodes.curves import pad_curve, linear_curve, move_curve, sample_array
from nodes.lights import sender
from nodes.primitives import color_array
from nodes.timing import seconds, SpeedMaster
from nodes.color import set_brightness_array, set_brightness_all, set_saturation_all
from nodes.lights import light_strip
from nodes.devices import get_gamepad_state, get_gain
from datatypes import ColorArray, Array
import jax.numpy as jnp
import jax

limiter = FrameLimiter(30)
speed_master1 = SpeedMaster()
color1 = 0
color2 = 0
noise_amount = 0.0
latch1 = False
latch2 = False

while True:
    out = [None, None, None]
    gamepad_state = get_gamepad_state()
    # if jnp.linalg.norm(gamepad_state[-1]) > 0.4:
    #     color1 = (
    #         jnp.arctan2(gamepad_state[-1][1], gamepad_state[-1][0]) % (2 * jnp.pi)
    #     ) / (2 * jnp.pi)
    # if jnp.linalg.norm(gamepad_state[-2]) > 0.4:
    #     color2 = (
    #         jnp.arctan2(gamepad_state[-2][1], gamepad_state[-2][0]) % (2 * jnp.pi)
    #     ) / (2 * jnp.pi)

    # if gamepad_state[12] == 1 and not latch1:
    #     latch1 = True
    #     speed_master1.speed_up()
    # elif gamepad_state[12] == 0:
    #     latch1 = False

    # if gamepad_state[13] == 1 and not latch2:
    #     latch2 = True
    #     speed_master1.slow_down()
    # elif gamepad_state[13] == 0:
    #     latch2 = False

    # if gamepad_state[1] == 1:
    #     noise_amount = 1.0
    # if not gamepad_state[0]:
    #     noise = jax.random.uniform(
    #         jax.random.PRNGKey(int(seconds() * 100)), shape=(100,)
    #     )
    #     output = set_brightness_array(
    #         color_array(100),
    #         sample_array(
    #             move_curve(
    #                 pad_curve(linear_curve(), 0.5),
    #                 1 - speed_master1.time,
    #             ),
    #             100,
    #         ),
    #     )
    #     output = output.at[2, :].max(noise * noise_amount)
    #     out[0] = output.at[1, :].set(1 - gamepad_state[-4]).at[0, :].set(color2)
    #     out[1] = output.at[1, :].set(1).at[0, :].set(color1)
    #     out[2] = output.at[1, :].set(1 - gamepad_state[-3]).at[0, :].set(color2)
    # else:
    #     noise = jax.random.uniform(
    #         jax.random.PRNGKey(int(seconds() * 100)), shape=(300,)
    #     )
    #     output = set_brightness_array(
    #         color_array(300),
    #         sample_array(
    #             move_curve(
    #                 pad_curve(linear_curve(), 0.75),
    #                 1 - speed_master1.time,
    #             ),
    #             300,
    #         ),
    #     )
    #     output = output.at[2, :].max(noise * noise_amount)
    #     output = jnp.array_split(output.at[1, :].set(1).at[0, :].set(color1), 3, axis=1)
    #     # print(output[0].shape)
    #     out[0] = output[0]
    #     out[1] = output[1]
    #     out[2] = output[2]

    # if gamepad_state[2] == 1:
    #     if (speed_master1.time % (1/4)) * 4 > 0.5:
    #         output = set_brightness_array(
    #             color_array(100),
    #             1.0,
    #         )
    #         output = set_saturation_all(output, 0.0)
    #         out[0] = output
    #         out[1] = output
    #         out[2] = output
    #     else:
    #         output = set_brightness_array(
    #             color_array(100),
    #             0.0,
    #         )
    #         out[0] = output
    #         out[1] = output
    #         out[2] = output
    print(get_gain())
    out[0] = set_saturation_all(set_brightness_array(color_array(100), min(1, (get_gain()/80))/3), 0)
    out[1] = set_saturation_all(set_brightness_array(color_array(100), min(1, (get_gain()/80))/3), 0)
    out[2] = set_saturation_all(set_brightness_array(color_array(100), min(1, (get_gain()/80))/3), 0)

    light_strip(2, out[0])
    light_strip(3, out[1])
    light_strip(4, out[2])
    noise_amount -= 0.06
    speed_master1.increment()
    limiter.tick()
    sender.flush()

print("slo mimo")
