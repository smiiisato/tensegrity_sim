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
        model_path = self.rospack.get_path("tensegrity_sim") + "/model/scene.xml"
        # 5 : frame skip
        MujocoEnv.__init__(self, model_path, 5)

        utils.EzPickle.__init__(self)

    def set_param(self):
        # robot joint / pose id
        # index zero is ground
        self.robot_pose_indices = self.model.jnt_qposadr[1:]
        self.robot_vel_indices = self.model.jnt_dofadr[1:]

        # name to id 
        self.link1_id = self.model.joint_name2id("link1")
        self.link2_id = self.model.joint_name2id("link2")
        self.link3_id = self.model.joint_name2id("link3")
        self.link4_id = self.model.joint_name2id("link4")
        self.link5_id = self.model.joint_name2id("link5")
        self.link6_id = self.model.joint_name2id("link6")

        # sensor id
        self.gyro_id = self.model.sensor_name2id("gyro")
        self.accelerometer_id = self.model.sensor_name2id("accelerometer")

        # control range
        self.ctrl_max = [0]*12
        self.ctrl_min = [-0.5]*12

        self.n_prev = 6
        self.max_episode = 1000
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
        self.body_xpos_rand = np.array([0.01]*18)
        self.const_body_xpos_rand = np.array([0.01]*18)

        # data.qpos = 7 dof * 6 links = 42
        # data.qvel = 6 dof * 6 links = 36

        if self.test:
            self.default_step_rate = 0.5

        # variable for rl
        self.const_ext_qpos = np.zeros(42)
        self.const_ext_force = np.zeros(36)
        self.const_ext_action = np.zeros(12)
        self.current_qpos = None
        self.current_qvel = None # not used
        #self.current_bvel = None
        self.current_body_xpos = None
        self.prev_qpos = None
        self.prev_qvel = None # not used
        #self.prev_bvel = None
        self.prev_body_xpos = None
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
                
        if self.current_body_xpos is None:
            self.current_body_xpos = self.sim.data.body_xpos.flat[3:]

        # save previous data
        if self.prev_action is None:
            self.prev_action = [copy.deepcopy(action) for i in range(self.n_prev)]

        if self.prev_qpos is None:
            self.prev_qpos = [copy.deepcopy(self.current_qpos) for i in range(self.n_prev)]

        if self.prev_qvel is None:
            self.prev_qvel = [copy.deepcopy(self.current_qvel) for i in range(self.n_prev)]

        if self.prev_body_xpos is None:
            self.prev_body_xpos = [copy.deepcopy(self.current_body_xpos) for i in range(self.n_prev)]

        # add random noise
        action_rate = 1.0 + self.action_rand*step_rate*np.random.randn(12) + self.const_ext_action
        action_converted = [(cmin+(rate*a+1.0)*(cmax-cmin)/2.0) for a, cmin, cmax, rate in zip(action, self.ctrl_min, self.ctrl_max, action_rate)] # tension force (12)
                
        self.sim.data.qfrc_applied[:] = self.const_ext_force + self.force_rand*step_rate*np.random.randn(36) # body linear force [N] + torque = 36 dof

        # do simulation
        self.do_simulation(action_converted, self.frame_skip)
        # print([self.sim.data.sensordata[self.model.sensor_adr[self.model.sensor_name2id(name)]] for name in ["dA_top", "dB_top", "dC_top", "dA_bottom", "dB_bottom", "dC_bottom"]])
            
        # next state without noise to calculate reward
        pose = self.sim.data.qpos[self.robot_pose_indices] # joint angle
        vel = self.sim.data.qvel[self.robot_vel_indices] # joint velocity
            # quat = self.sim.data.qpos[[4, 5, 6, 3]] # [x, y, z, w]
            # pole_quat = self.sim.data.body_xquat[self.model.nbody-2][[1, 2, 3, 0]]

        self.current_qpos = self.sim.data.qpos + self.qpos_rand*step_rate*np.random.randn(42) + self.const_ext_qpos
        self.current_qvel = self.sim.data.qvel + self.qvel_rand*step_rate*np.random.randn(36)
        self.current_body_xpos = self.sim.data.body_xpos.flat[3:] + self.body_xpos_rand*step_rate*np.random.randn(18)

        # reward definition
        forward_reward = 0.0
        ctrl_reward = 0.0
        rotate_reward = 0.0
        #rotate_speed_reward = 0.0
        #yaw_reward = 0.0
        survive_reward = 0.0
        
        # diff of body_xpos
        forward_value = 1.0*(
                    (self.current_body_xpos[0]-self.prev_body_xpos[-1][0])
                    +(self.current_body_xpos[3]-self.prev_body_xpos[-1][3])
                    +(self.current_body_xpos[6]-self.prev_body_xpos[-1][6])
                    +(self.current_body_xpos[9]-self.prev_body_xpos[-1][9])
                    +(self.current_body_xpos[12]-self.prev_body_xpos[-1][12])
                    +(self.current_body_xpos[15]-self.prev_body_xpos[-1][15])
                    )
        
        """
        forward_reward
        """
        if forward_value > 0:
            forward_reward = 1.0*forward_value**2
        else:
            forward_reward = -1.0*step_rate*forward_value

        ctrl_reward = -1.0*step_rate*np.linalg.norm(action)
        #print("action:{}".format(np.linalg.norm(action)))
        #print("xpos:{}".format(self.current_body_xpos[0]))
        #print("gyro:{}".format(self.sim.data.sensordata[1:4]))
        #print("qvel:{}".format(self.sim.data.qvel))
        #print("qvel:{}".format(np.square(self.sim.data.qvel[3:5]).sum()))
        """
        rotate_reward = 0.1*step_rate*(
                np.square(self.sim.data.qvel[3:5]).sum()+
                np.square(self.sim.data.qvel[9:11]).sum()+
                np.square(self.sim.data.qvel[15:17]).sum()+
                np.square(self.sim.data.qvel[21:23]).sum()+
                np.square(self.sim.data.qvel[27:29]).sum()+
                np.square(self.sim.data.qvel[33:35]).sum())
        rotate_speed_reward = -1.0*step_rate*np.linalg.norm(self.sim.data.sensordata[1:4])
        """
        # size of roll/pitch
        rotate_value = (
                np.square(self.sim.data.qvel[3:5]).sum()+
                np.square(self.sim.data.qvel[9:11]).sum()+
                np.square(self.sim.data.qvel[15:17]).sum()+
                np.square(self.sim.data.qvel[21:23]).sum()+
                np.square(self.sim.data.qvel[27:29]).sum()+
                np.square(self.sim.data.qvel[33:35]).sum())
        if rotate_value < 600:
            rotate_reward = 0.0001*rotate_value
        else:
            rotate_reward = -0.0001*step_rate*rotate_value

        # size of yaw
        """
        yaw_reward = -0.1*step_rate*(
                np.square(self.sim.data.qvel[5])+
                np.square(self.sim.data.qvel[11])+
                np.square(self.sim.data.qvel[17])+
                np.square(self.sim.data.qvel[23])+
                np.square(self.sim.data.qvel[29])+
                np.square(self.sim.data.qvel[35])
                )
        """
        print("rotate_value:{}".format(rotate_reward))
        print("forward_value:{}".format(forward_reward))
        print("ctrl_reward:{}".format(ctrl_reward))
        survive_reward = 0.1
        reward = ctrl_reward + forward_reward + rotate_reward + survive_reward
#        print("ctrl:{}".format(ctrl_reward))
#        print("rotate:{}".format(rotate_reward))

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
        
        # update prev data
        self.prev_qvel.append(copy.deepcopy(self.current_qvel))
        #self.prev_body_xpos = self.prev_body_xpos.tolist()
        self.prev_body_xpos.append(copy.deepcopy(self.current_body_xpos))
        if len(self.prev_qpos) > self.n_prev:
            del self.prev_qpos[0]
        if len(self.prev_qvel) > self.n_prev:
            del self.prev_qvel[0]
        if len(self.prev_body_xpos) > self.n_prev:
            del self.prev_body_xpos[0]
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
                rotate_reward = rotate_reward,
                ctrl_reward = ctrl_reward,
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
            ]
        )

    def _set_action_space(self):
        low = np.asarray([-0.5]*12, dtype=np.float32)
        high = np.asarray([0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0], dtype=np.float32)
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
            self.current_body_xpos = self.sim.data.body_xpos.flat[3:]
            self.prev_action = [np.zeros(12) for i in range(self.n_prev)]
            self.prev_qpos = [self.current_qpos + self.qpos_rand*step_rate*np.random.randn(42) for i in range(self.n_prev)]
            self.prev_qvel = [self.current_qvel + self.qvel_rand*step_rate*np.abs(self.sim.data.qvel)*np.random.randn(36) for i in range(self.n_prev)]
            self.prev_body_xpos = [self.current_body_xpos + self.body_xpos_rand*step_rate*np.abs(self.sim.data.body_xpos.flat[3:])*np.random.randn(18) for i in range(self.n_prev)]

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0
