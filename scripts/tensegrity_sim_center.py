import copy
import numpy as np
from rospkg import RosPack
from gym import utils, spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv


class TensegrityEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, test=False, ros=False, max_step=None):
        self.is_params_set = False
        self.test = test
        self.ros = ros
        self.max_step = max_step

        if self.test and self.ros:
            import rospy
            from std_msgs.msg import Float32MultiArray
            rospy.init_node("tensegrity_env")
            self.debug_msg = Float32MultiArray()
            self.debug_pub = rospy.Publisher("tensegrity_env/debug", Float32MultiArray, queue_size=10)

        self.rospack = RosPack()
        model_path = self.rospack.get_path("tensegrity_sim") + "/model/scene_24act.xml"
        # 5 : frame skip
        MujocoEnv.__init__(self, model_path, 5)

        utils.EzPickle.__init__(self)

    def set_param(self):

        # sensor id
        self.gyro_id = self.model.sensor_name2id("gyro")

        # control range
        self.ctrl_max = [0]*24
        self.ctrl_min = [-6.0]*24

        self.n_prev = 6
        self.max_episode = 3000
        #self.max_episode = 1000

        # random noise
        self.qpos_rand = np.array([0.01]*42, dtype=np.float32) # 7dof * 6 links = 42
        self.const_qpos_rand = np.array([0.01]*42, dtype=np.float32) # 7dof * 6 links = 42
        self.qvel_rand = np.array([
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            0.01, 0.01, 0.01, 0.03, 0.03, 0.01,
            ], dtype=np.float32) # velocity of quaternion (3) + joint velocities (3) = (6)
        self.force_rand = np.array([0.003]*36, dtype=np.float32)
        self.const_force_rand = np.array([0.003]*36, dtype=np.float32)
        #self.torque_rand = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        #self.const_torque_rand = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.action_rand = np.array([0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        self.const_action_rand = np.array([0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        self.get_body_xpos_rand = np.array([0.01]*18)
        self.const_get_body_xpos_rand = np.array([0.01]*18)
        # data.qpos = 7 dof * 6 links = 42
        # data.qvel = 6 dof * 6 links = 36

        if self.test:
            self.default_step_rate = 0.5

        # variable for rl
        self.const_ext_qpos = np.zeros(42)
        self.const_ext_force = np.zeros(36)
        self.const_ext_action = np.zeros(24)
        self.current_qpos = None
        self.current_qvel = None # not used
        #self.current_bvel = None
        self.current_get_body_xpos = None
        self.prev_qpos = None
        self.prev_qvel = None # not used
        #self.prev_bvel = None
        self.prev_get_body_xpos = None
        self.prev_action = None
        self.episode_cnt = 0
        self.step_cnt = 0

    def step(self, action): # action : tension of each tendon, 12dof
        if not self.is_params_set:
            self.set_param()
            self.is_params_set = True

        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        # quarternion (4) + joint angles of slide/roll/pitsh (3) = (7)
        if self.current_qpos is None:
            self.current_qpos = self.sim.data.qpos

        # velocity of quarternion (3) + joint velocity of roll/pitch/slide (3) = (6)
        if self.current_qvel is None:
            self.current_qvel = self.sim.data.qvel
                
        if self.current_get_body_xpos is None:
            self.current_get_body_xpos = np.hstack((
                    self.sim.data.get_body_xpos("link1"),
                    self.sim.data.get_body_xpos("link2"),
                    self.sim.data.get_body_xpos("link3"),
                    self.sim.data.get_body_xpos("link4"),
                    self.sim.data.get_body_xpos("link5"),
                    self.sim.data.get_body_xpos("link6")
                    ))

        # save previous data
        if self.prev_action is None:
            self.prev_action = [copy.deepcopy(action) for i in range(self.n_prev)]

        if self.prev_qpos is None:
            self.prev_qpos = [copy.deepcopy(self.current_qpos) for i in range(self.n_prev)]

        if self.prev_qvel is None:
            self.prev_qvel = [copy.deepcopy(self.current_qvel) for i in range(self.n_prev)]

        if self.prev_get_body_xpos is None:
            self.prev_get_body_xpos = [copy.deepcopy(self.current_get_body_xpos) for i in range(self.n_prev)]

        #action_rate = 1.0 + self.action_rand*step_rate*np.random.randn(12) + self.const_ext_action
        #action_converted = [(cmin+(rate*a+1.0)*(cmax-cmin)/2.0) for a, cmin, cmax, rate in zip(action, self.ctrl_min, self.ctrl_max, action_rate)] # tension force (12)
        print("action:{}".format(action))

        # do simulation
        self.do_simulation(action, self.frame_skip)
            
        # reward definition
        forward_reward = 0.0
        ctrl_reward = 0.0
       
        # global frame position
        forward_reward = 1.0*(
                sum(self.current_get_body_xpos[::3])/6
                    )
        
        ctrl_reward = -0.005*step_rate*np.linalg.norm(action)
        reward = forward_reward + ctrl_reward
        #print("ctrl:{}".format(ctrl_reward))
        #print("forward:{}".format(forward_reward))
        print(sum(self.current_get_body_xpos[::3])/6)

        if self.test and self.ros:
            self.debug_msg.data = np.concatenate([np.array(action_converted), pose, vel, self.sim.data.qvel])
            self.debug_pub.publish(self.debug_msg)

        self.episode_cnt += 1
        self.step_cnt += 1
        
        #notdone = np.square(self.sim.data.qvel).sum() < 2000
        notdone = self.episode_cnt < self.max_episode
        
        #notdone = self.episode_cnt < self.max_episode
        if self.step_cnt == 1:
            done = False
        else:
            done = not notdone
            
        self.current_qpos = self.sim.data.qpos
        self.current_qvel = self.sim.data.qvel 
        self.current_get_body_xpos = np.hstack((
                    self.sim.data.get_body_xpos("link1"),
                    self.sim.data.get_body_xpos("link2"),
                    self.sim.data.get_body_xpos("link3"),
                    self.sim.data.get_body_xpos("link4"),
                    self.sim.data.get_body_xpos("link5"),
                    self.sim.data.get_body_xpos("link6")
                    ))
        self.prev_qpos.append(copy.deepcopy(self.current_qpos))
        self.prev_qvel.append(copy.deepcopy(self.current_qvel))
        self.prev_get_body_xpos.append(copy.deepcopy(self.current_get_body_xpos))
        if len(self.prev_qpos) > self.n_prev:
            del self.prev_qpos[0]
        if len(self.prev_qvel) > self.n_prev:
            del self.prev_qvel[0]
        if len(self.prev_get_body_xpos) > self.n_prev:
            del self.prev_get_body_xpos[0]
        obs = self._get_obs()
        self.prev_action.append(copy.deepcopy(action))
        if len(self.prev_action) > self.n_prev:
            del self.prev_action[0]
        if done:
            self.episode_cnt = 0
            self.current_qpos = None
            self.current_qvel = None
            self.prev_action = None
            self.prev_qpos = None
            self.prev_qvel = None
            self.const_ext_qpos = self.const_qpos_rand*step_rate*np.random.randn(42)
            self.const_ext_force = self.const_force_rand*step_rate*np.random.randn(36)
            self.const_ext_action = self.const_action_rand*step_rate*np.random.randn(12)
        return (
            obs,
            reward,
            done,
            dict(
                forward_reward = forward_reward,
                ),
            )


    def _get_obs(self):
        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate
        return np.concatenate(
            [
                np.concatenate(self.prev_qpos), # prev base quat + joint angles
                np.concatenate(self.prev_qvel), # prev base quat vel + joint vels
                np.concatenate(self.prev_action), # prev action
                np.concatenate(self.prev_get_body_xpos), 
            ]
        )

    def _set_action_space(self):
        low = np.asarray([-6.0]*24, dtype=np.float32)
        high = np.asarray([0]*24, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space


    def reset_model(self):
        # joint id

        if self.max_step:
            step_rate = float(self.step_cnt)/self.max_step
        elif self.test:
            step_rate = self.default_step_rate

        qpos = self.init_qpos
        # define initial joint angle
        qpos = np.array([-0.1, 0, 0, 1.0, 0, 0, 0,
                0.1, 0, 0, 1.0, 0, 0, 0,
                0, 0.1, 0, 1.0, 0, 0, 0, 
                0, -0.1, 0, 1.0, 0, 0, 0,
                0, 0, 0.1, 1.0, 0, 0, 0,
                0, 0, -0.1, 1.0, 0, 0, 0
                ])
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        if (self.prev_qpos is None) and (self.prev_action is None):
            self.current_qpos = self.sim.data.qpos.flat[:]
            self.current_qvel = self.sim.data.qvel.flat[:]
            self.current_get_body_xpos = np.hstack((
                    self.sim.data.get_body_xpos("link1"),
                    self.sim.data.get_body_xpos("link2"),
                    self.sim.data.get_body_xpos("link3"),
                    self.sim.data.get_body_xpos("link4"),
                    self.sim.data.get_body_xpos("link5"),
                    self.sim.data.get_body_xpos("link6")
                    ))
            self.prev_action = [np.zeros(24) for i in range(self.n_prev)]
            self.prev_qpos = [self.current_qpos for i in range(self.n_prev)]
            self.prev_qvel = [self.current_qvel for i in range(self.n_prev)]
            self.prev_get_body_xpos = [self.current_get_body_xpos for i in range(self.n_prev)]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0
