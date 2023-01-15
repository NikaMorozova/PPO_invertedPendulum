import numpy as np
import mujoco
import mujoco.viewer
from pyvirtualdisplay import Display
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

class InvertedPendulumEnv:
    xml_env = """
    <mujoco model="inverted pendulum">
            <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="160" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        </asset>
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="1" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.0 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-3 3"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule" group="3"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-100000 100000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
        </actuator>
    </mujoco>
    """

    def __init__(
        self,
        target = False,
        upswing = False,
        mass = None,
        test = True
    ):
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
        self.pos_ball = np.zeros(1)
        self.position = [0, 0, 0]
        self.model = mujoco.MjModel.from_xml_string(InvertedPendulumEnv.xml_env)
        self.data = mujoco.MjData(self.model)
        self.observation_space=Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space=Box(-3.0, 3.0, (1,), dtype=np.float32)
        self.model.opt.timestep = 0.02
        self.init_m = self.model.body_mass.copy()
        self.init_i = self.model.body_inertia.copy()
        self.target = target
        self.upswing = upswing
        self.test = test
        self.last_update = 0
        self.mass = mass
        if self.test:
            self.frames = []
            self.width, self.height = 640, 480
            self.display = Display(visible=0, size=(self.width, self.height))
            self.display.start()
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            print(f"Renderer context created with dimensions: {self.width}x{self.height}")
        self.reset_model()


        """
        This part for additionally tested reward
        """
        # self.velocity_previous = 0
        # self.dist_to_vertical_previous = None
        # self.previous_obs = None

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        if self.test:
            self.renderer.update_scene(self.data)
            if self.target and self.data.time - self.last_update > 5:
                self.last_update = self.data.time
                self.position = [np.random.rand() - 0.5, 0, 0.6]
            self.draw_ball(self.position, radius=0.05)
            pixels = self.renderer.render()
            self.frames.append(pixels)

        obs = self.obs()
        rewards = self.reward(obs)

        terminated = bool(np.abs(obs[1]) > 0.15)
        truncated = False
        if self.current_time > 50:
            truncated = True
        return obs, rewards, terminated, truncated

    def obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        #concatenate
        if self.target:
            res = np.concatenate([qpos, qvel, self.pos_ball.ravel()])

        else:
            res = np.concatenate([qpos, qvel]).ravel()
        return res

    def reset_model(self):
        self.data.qpos = self.init_qpos
        self.data.qvel = self.init_qvel

        self.data.qpos[1] = -np.pi

        self.pos_ball = np.array(self.position[0])

        if self.mass is not None:
            base = np.array([[1.0], [1.0], [self.mass]])
        else:
            base = np.array([[1.0], [1.0], [np.random.uniform(0.1, 10.0)]])
        const = base * np.ones((1, 3))
        self.model.body_mass = self.init_m * const[:,0]
        self.model.body_inertia = const * self.init_i

        self.data.time = 0.0
        self.data.ctrl = 0.0
        self.last_update = 0.0

        mujoco.mj_step(self.model, self.data)

        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def draw_ball(self, position, color=[1, 0, 0, 1], radius=0.01):
        if self.target:
            self.pos_ball = np.array([self.position[0]])
        if self.renderer.scene.ngeom >= self.renderer.scene.maxgeom:
          return
        mujoco.mjv_initGeom(self.renderer.scene.geoms[self.renderer.scene.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[radius, 0, 0],
                            pos=np.array(self.position),
                            mat=np.eye(3).flatten(),
                            rgba=np.array(color))
        self.renderer.scene.ngeom += 1  # increment ngeom

    @property
    def current_time(self):
        return self.data.time


    def reward(self, ob, len_pole=0.6):
        if abs(ob[1]) < 0.15:
            if np.abs(ob[0] - ob[4] < 0.1):
                alpha = 1
            else: alpha = -1
            reward = np.cos(ob[1]) + np.exp(-np.abs(ob[0] - ob[4])) * alpha
        else:
            reward = np.cos(ob[1]) - 0.00001 * ob[3]**2 - 0.01 * ob[0]**2 - 0.00001 * ob[2]**2
        return reward
    """
    This reward was also explored in experiments, but simplicity is the best match for this task
    """
    # def reward(self, obs, a, target, len_pole=0.6):
    #     #Here we firstly compute reward's elements

    #     #Penalty for get out of range
    #     if abs(obs[0]) >= 1.0:
    #         hit_penalty = 1.0
    #     else:
    #         hit_penalty = 0
        
    #     #Gravity
    #     theta = obs[1]
    #     gravity_moment = - self.model.body_mass * 9.81 * len_pole * np.sin(theta)

    #     pole_end_point = [obs[0] + np.sin(theta) * len_pole, 0, np.cos(theta) * len_pole]
        
    #     #Distance to target
    #     dist_to_target = np.linalg.norm(np.array(self.pos_ball) - np.array(pole_end_point))

    #     #Gravitation bonus
    #     if dist_to_vertical < 0.10 * np.pi * len_pole:
    #         bonus_gravity =  (0.5 ** 2) / (1 + abs(gravity_moment) * 100 * dist_to_target)   
    #     else: bonus_gravity = 0

    #     #Distance to vertical
    #     z = pole_end_point[2]
    #     theta = np.arccos(np.clip(z / len_pole, -1.0, 1.0))  # angle from vertical
    #     dist_to_vertical = len_pole * theta

    #     #Velocity of the pole
    #     vx = len_pole * obs[3] * np.cos(theta) #for tip only    
    #     vz = len_pole * obs[3] * np.sin(theta)

    #     velocity_magnitude = np.sqrt(vx**2 + vz**2)

    #     vx = obs[2] + len_pole * obs[3] * np.cos(theta) #for tip and cart

    #     velocity = np.sqrt(vx**2 + vz**2)
        
    #     #Small dist to target bonus
    #     if dist_to_vertical < 0.10 * np.pi * len_pole:
    #         dist_to_target_bonus = 0.5 / (1 + velocity * 100 * dist_to_target)
    #     else: dist_to_target_bonus = 0

    #     #Bonus to put tip vertical
    #     theta = dist_to_vertical / len_pole
    #     ideal_velocity = (1 - np.cos(theta))
    #     error = velocity_magnitude - ideal_velocity
    #     bonus_to_vertical = np.exp(-np.sqrt(abs(error)))

    #     bonus =  np.cos(obs[1]/2) *(bonus_to_vertical) / 2 +\
    #                                      (bonus_gravity +  dist_to_target_bonus) / 2
    #     reward = bonus - hit_penalty

    #     self.velocity_previous = velocity_magnitude
    #     self.dist_to_vertical_previous = dist_to_vertical
    #     return reward