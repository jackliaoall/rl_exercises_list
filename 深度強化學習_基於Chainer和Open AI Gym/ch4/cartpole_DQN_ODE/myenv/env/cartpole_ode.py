# -*- coding: utf-8 -*-
"""
倒立振子の動作プログラム（ODE使用）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import ode

logger = logging.getLogger(__name__)

world = ode.World()
world.setGravity((0,-9.81,0))

body1 = ode.Body(world)
M = ode.Mass()
M.setBox(250, 1, 0.5, 0.1)
M.mass = 1.0
body1.setMass(M)
body1.setPosition((0,0,0))
body1.setForce((1,0,0))
body2 = ode.Body(world)
M = ode.Mass()
M.setBox(2.5, 0.2, 2, 0.2)
M.mass = 0.1
body2.setMass(M)
body2.setPosition((0,0.5,0))

j1 = ode.SliderJoint(world)
j1.attach(body1, ode.environment)
j1.setAxis( (1,0,0) )

j2 = ode.HingeJoint(world)
j2.attach(body1, body2)
j2.setAnchor( (0,0,0) )
j2.setAxis( (0,0,1) )

class CartPoleODEEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.force_mag = 10.0
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        force = self.force_mag if action==1 else -self.force_mag
        body1.setForce((force,0,0))
        world.step(0.02)

        x = body1.getPosition()[0]
        v = body1.getLinearVel()[0]
        a = math.asin(body2.getRotation()[1])
        w = body2.getAngularVel()[2]

        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or a < -self.theta_threshold_radians \
                or a > self.theta_threshold_radians
        done = bool(done)

        reward = 0.0
        if not done:
            reward = 1.0
            
        self.state = (x,v,a,w)

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = (0,0,0,0)
        body1.setPosition((0,0,0))
        body1.setLinearVel((0,0,0))
        body1.setForce((1,0,0))
        body2.setPosition((0,0.5,0))
        body2.setLinearVel((0,0,0))
        body2.setForce((0,0,0))
        body2.setQuaternion((1,0,0,0))
        body2.setAngularVel((1,0,0,0))
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * 1.0 *2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            l = 1/2*scale
            h = 0.5/2*scale
            ball1 = rendering.FilledPolygon([(-l,h), (l,h), (l,-h), (-l,-h)])
            self.balltrans1 = rendering.Transform()
            ball1.add_attr(self.balltrans1)
            ball1.set_color(.5,.5,.5)
            self.viewer.add_geom(ball1)
            
            l = 1*scale
            h = 0.1/2*scale
            ball2 = rendering.FilledPolygon([(0,h), (l,h), (l,-h), (0,-h)])
            self.balltrans2 = rendering.Transform(translation=(0, 0))
            ball2.add_attr(self.balltrans2)
            ball2.add_attr(self.balltrans1)
            ball2.set_color(0,0,0)
            self.viewer.add_geom(ball2)

        if self.state is None: return None

        x1,y1,z1 = body1.getPosition()
#        self.balltrans1.set_translation(x1*scale+screen_width/2.0, 0*scale+screen_height/2.0)
        self.balltrans1.set_translation(x1*scale+screen_width/2.0, 0*scale+screen_height/2.0)
        self.balltrans2.set_rotation(math.asin(body2.getRotation()[1])+3.14/2)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
