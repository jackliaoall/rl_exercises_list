# -*- coding: utf-8 -*-
"""
リフティングの動作プログラム（ODE使用）
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
world.setGravity( (0,-9.81,0) )

body1 = ode.Body(world)
M = ode.Mass()
M.setBox(250, 1.0, 0.2, 0.1)
M.mass = 1.0
body1.setMass(M)
body2 = ode.Body(world)
M = ode.Mass()
M.setSphere(25.0, 0.1)
M.mass = 0.01
body2.setMass(M)

j1 = ode.SliderJoint(world)
j1.attach(body1, ode.environment)
j1.setAxis( (1,0,0) )

space = ode.Space()
Racket_Geom = ode.GeomBox(space, (1.0, 0.2, 0.1))
Racket_Geom.setBody(body1)
Ball_Geom = ode.GeomSphere(space, radius=0.05)
Ball_Geom.setBody(body2)
contactgroup = ode.JointGroup()

Col = False

def Collision_Callback(args, geom1, geom2):
    contacts = ode.collide(geom1, geom2)
    world, contactgroup = args
    for c in contacts:
        c.setBounce(1) # 反発係数
        c.setMu(0)  # クーロン摩擦係数
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())
        global Col
        Col=True

class LiftingODEEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.Col = False
        self.gravity = 9.8
        self.cartmass = 1.0
        self.cartwidth = 1.0#0.5#2# 1#
        self.carthdight = 0.2
        self.cartposition = 0
        self.ballPosition = 1#2.4
        self.ballRadius = 0.1#2.4
        self.ballVelocity = 1
        self.force_mag = 10.0
        self.tau = 0.01  # seconds between state updates

        self.cx_threshold = 2.4
        self.bx_threshold = 2.4
        self.by_threshold = 2.4

        high = np.array([
            self.cx_threshold,
            np.finfo(np.float32).max,
            self.bx_threshold,
            self.by_threshold,
            np.finfo(np.float32).max
            ])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        force = self.force_mag if action==1 else -self.force_mag
        reward = 0.0
        space.collide((world, contactgroup), Collision_Callback)
        body1.setForce( (force,0,0) )
        world.step(self.tau)
        contactgroup.empty()
        bx,by,bz = body2.getPosition()
        bu,bv,bw = body2.getLinearVel()
        rx,ry,rz = body1.getPosition()
        ru,rv,rw = body1.getLinearVel()
        self.state = (rx,ru,bx,by,bu)
        done =  by < -0.2
        done = bool(done)
        
        global Col
        if Col:
            Col = False
            reward = 1.0

        if bx > self.bx_threshold or bx < -self.bx_threshold:
            body2.setLinearVel((-bu, bv, bw))
        
        return np.array(self.state), reward, done, {}
    def _reset(self):
        body1.setPosition((0,0,0))
        body1.setLinearVel((0,0,0))
        body1.setForce((1,0,0))
        body2.setPosition((0,self.ballPosition,0))
        body2.setLinearVel((self.ballVelocity,0,0))
        body2.setForce((0,0,0))
        Col = False

        rx,ry,rz = body1.getPosition()
        ru,rv,rw = body1.getLinearVel()
        bx,by,bz = body2.getPosition()
        bu,bv,bw = body2.getLinearVel()
        self.state = (rx,ru,bx,by,bu)
        self.steps_beyond_done = None
        self.by_dot = 0
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        world_width = self.cx_threshold*2
        scale = screen_width/world_width
        cartwidth = self.cartwidth*scale#50.0
        cartheight = 30.0


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l = self.cartwidth/2*scale
            h = self.carthdight/2*scale
            ball1 = rendering.FilledPolygon([(-l,h), (l,h), (l,-h), (-l,-h)])
            self.balltrans1 = rendering.Transform()
            ball1.add_attr(self.balltrans1)
            ball1.set_color(.5,.5,.5)
            self.viewer.add_geom(ball1)
            
            ball2 = rendering.make_circle(self.ballRadius*scale)
            self.balltrans2 = rendering.Transform(translation=(0, 0))
            ball2.add_attr(self.balltrans2)
            ball2.set_color(0,0,0)
            self.viewer.add_geom(ball2)

        if self.state is None: return None

        x1,y1,z1 = body1.getPosition()
        x2,y2,z2 = body2.getPosition()
        self.balltrans1.set_translation(x1*scale+screen_width/2.0, 0*scale+screen_height/2.0)
        self.balltrans2.set_translation(x2*scale+screen_width/2.0, y2*scale+screen_height/2.0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
