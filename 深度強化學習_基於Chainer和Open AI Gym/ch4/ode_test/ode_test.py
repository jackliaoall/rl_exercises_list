# -*- coding: utf-8 -*-
"""
ODEのテスト用プログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import ode

world = ode.World()
world.setGravity( (0,-9.81,0) )

body = ode.Body(world)
M = ode.Mass()
M.setSphere(2500.0, 0.05)
M.mass = 1.0
body.setMass(M)

body.setPosition( (0,2,0) )
body.setLinearVel( (20,10,0) )

total_time = 0.0
dt = 0.01
while total_time<3.0:
    x,y,z = body.getPosition()
    u,v,w = body.getLinearVel()
    print(total_time,'\t', x,'\t', y,'\t', z,'\t', u,'\t',v,'\t', w,sep='')
    world.step(dt)
    total_time+=dt
