#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from std_msgs.msg import Float64MultiArray
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

P1,P2,P3,Pc,Pr,Pb,thetac,A,b = None,None,None,None,None,None,None,None,None
camera_cmd_vel = Twist()
ranging_cmd_vel = Twist()
bearing_cmd_vel = Twist()
fx,fy,lx,ly = 0.1496485702,0.1496485702,0.1693333333,0.127
#fx,fy,lx,ly = 565.6,565.6,640,480
sigma_u,sigma_v,sigma_ranging,sigma_bearing,sigma_alpha = 0.007,0.007,0.01,0.01,0.01
#sigma_u,sigma_v,sigma_ranging,sigma_bearing,sigma_alpha = 1,1,1,1,1
x_fov_wealth = 3*pi/180
y_fov_wealth = 3*pi/180
height_l = 0.3
height_u = 1.8
d_safe_car = 0.7
d_measuring = 2.2
d_safe_uav = 0.7
d_communication = 20
gamma = 1.0
time_last,dt = 0,0

class Objective(ElementwiseProblem):

	def __init__(self):
		super().__init__(n_var=10,n_obj=3,n_constr=b.size, \
						 xl=np.array([-0.3]*10),xu=np.array([0.3]*10))

	def _evaluate(self, x, out, *args, **kwargs):
		'''
		R = np.array([[sigma_u,0,0,0,0],[0,sigma_v,0,0,0],[0,0,sigma_ranging,0,0],[0,0,0,sigma_bearing,0],[0,0,0,0,sigma_alpha]])

		dO1 = np.array([ \
						[fx*(P1[1] - (Pc[1] + dt*x[1]))/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**2, -fx*(P1[0] - (Pc[0] + dt*x[0]))/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**2, 0], \
						[fy*cos(thetac + dt*x[9])*(P1[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**2, fy*sin(thetac + dt*x[9])*(P1[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**2, -fy/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))], \
						[(P1[0] - (Pr[0] + dt*x[3]))/np.linalg.norm(P1 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P1[1] - (Pr[1] + dt*x[4]))/np.linalg.norm(P1 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P1[2] - (Pr[2] + dt*x[5]))/np.linalg.norm(P1 - (Pr + dt*np.array([x[3],x[4],x[5]])))], \
						[-(P1[1] - (Pb[1] + dt*x[7]))/((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2), (P1[0] - (Pb[0] + dt*x[6]))/((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2), 0], \
						[-(P1[0] - (Pb[0] + dt*x[6]))*(P1[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P1 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)), -(P1[1] - (Pb[1] + dt*x[7]))*(P1[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P1 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)), sqrt((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)/np.linalg.norm(P1 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2], \
					  ])

		dO2 = np.array([ \
						[fx*(P2[1] - (Pc[1] + dt*x[1]))/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**2, -fx*(P2[0] - (Pc[0] + dt*x[0]))/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**2, 0], \
						[fy*cos(thetac + dt*x[9])*(P2[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**2, fy*sin(thetac + dt*x[9])*(P2[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**2, -fy/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))], \
						[(P2[0] - (Pr[0] + dt*x[3]))/np.linalg.norm(P2 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P2[1] - (Pr[1] + dt*x[4]))/np.linalg.norm(P2 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P2[2] - (Pr[2] + dt*x[5]))/np.linalg.norm(P2 - (Pr + dt*np.array([x[3],x[4],x[5]])))], \
						[-(P2[1] - (Pb[1] + dt*x[7]))/((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2), (P2[0] - (Pb[0] + dt*x[6]))/((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2), 0], \
						[-(P2[0] - (Pb[0] + dt*x[6]))*(P2[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P2 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)), -(P2[1] - (Pb[1] + dt*x[7]))*(P2[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P2 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)), sqrt((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)/np.linalg.norm(P2 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2], \
					  ])

		dO3 = np.array([ \
						[fx*(P3[1] - (Pc[1] + dt*x[1]))/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**2, -fx*(P3[0] - (Pc[0] + dt*x[0]))/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**2, 0], \
						[fy*cos(thetac + dt*x[9])*(P3[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**2, fy*sin(thetac + dt*x[9])*(P3[2] - (Pc[2] + dt*x[2]))/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**2, -fy/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))], \
						[(P3[0] - (Pr[0] + dt*x[3]))/np.linalg.norm(P3 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P3[1] - (Pr[1] + dt*x[4]))/np.linalg.norm(P3 - (Pr + dt*np.array([x[3],x[4],x[5]]))), (P3[2] - (Pr[2] + dt*x[5]))/np.linalg.norm(P3 - (Pr + dt*np.array([x[3],x[4],x[5]])))], \
						[-(P3[1] - (Pb[1] + dt*x[7]))/((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2), (P3[0] - (Pb[0] + dt*x[6]))/((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2), 0], \
						[-(P3[0] - (Pb[0] + dt*x[6]))*(P3[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P3 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)), -(P3[1] - (Pb[1] + dt*x[7]))*(P3[2] - (Pb[2] + dt*x[8]))/(np.linalg.norm(P3 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2*sqrt((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)), sqrt((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)/np.linalg.norm(P3 - (Pb + dt*np.array([x[6],x[7],x[8]])))**2], \
					  ])
		'''
		'''
		f1 = np.trace(np.transpose(dO1).dot(np.linalg.inv(R)).dot(dO1))
		f2 = np.trace(np.transpose(dO2).dot(np.linalg.inv(R)).dot(dO2))
		f3 = np.trace(np.transpose(dO3).dot(np.linalg.inv(R)).dot(dO3))
		'''
		'''
		S1 = np.linalg.svd(dO1,full_matrices=False, compute_uv=False)
		S2 = np.linalg.svd(dO2,full_matrices=False, compute_uv=False)
		S3 = np.linalg.svd(dO3,full_matrices=False, compute_uv=False)
		
		f1 = (S1[0]*S1[1]*S1[2])**2
		f2 = (S2[0]*S2[1]*S2[2])**2
		f3 = (S3[0]*S3[1]*S3[2])**2
		'''
		'''
		nc = np.array([cos(thetac + dt*x[9]),sin(thetac + dt*x[9]),0])
		r1c = np.array([P1[0] - (Pc[0] + dt*x[0]),P1[1] - (Pc[1] + dt*x[1]),P1[2] - (Pc[2] + dt*x[2])])
		r1b = np.array([P1[0] - (Pb[0] + dt*x[6]),P1[1] - (Pb[1] + dt*x[7]),P1[2] - (Pb[2] + dt*x[8])])

		r2c = np.array([P2[0] - (Pc[0] + dt*x[0]),P2[1] - (Pc[1] + dt*x[1]),P2[2] - (Pc[2] + dt*x[2])])
		r2b = np.array([P2[0] - (Pb[0] + dt*x[6]),P2[1] - (Pb[1] + dt*x[7]),P2[2] - (Pb[2] + dt*x[8])])

		r3c = np.array([P3[0] - (Pc[0] + dt*x[0]),P3[1] - (Pc[1] + dt*x[1]),P3[2] - (Pc[2] + dt*x[2])])
		r3b = np.array([P3[0] - (Pb[0] + dt*x[6]),P3[1] - (Pb[1] + dt*x[7]),P3[2] - (Pb[2] + dt*x[8])])


		f1_b = 1/np.linalg.norm(r1b[:2])**2
		f1_a = 1/r1b[2]**2
		f1_uv = r1c[2]**2/(nc.dot(r1c))**2

		f2_b = 1/np.linalg.norm(r2b[:2])**2
		f2_a = 1/r2b[2]**2
		f2_uv = r2c[2]**2/(nc.dot(r2c))**2

		f3_b = 1/np.linalg.norm(r3b[:2])**2
		f3_a = 1/r3b[2]**2
		f3_uv = r3c[2]**2/(nc.dot(r3c))**2
		
		out["F"] = [f1_b, f1_a, f1_uv, f2_b, f2_a, f2_uv, f3_b, f3_a, f3_uv]
		'''
		
		f1 = ( \
                         sigma_bearing**2*sigma_alpha**2*fx**2*fy**2*(P1[0]*(Pr[0] + dt*x[3]) + (Pc[0] + dt*x[0])*(P1[0] - (Pr[0] + dt*x[3])) + P1[1]*(Pr[1] + dt*x[4]) + (Pc[1] + dt*x[1])*(P1[1] - (Pr[1] + dt*x[4])) + P1[2]*(Pr[2] + dt*x[5]) + (Pc[2] + dt*x[2])*(P1[2] - (Pr[2] + dt*x[5])) - P1[0]**2 - P1[1]**2 - P1[2]**2)**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**6*((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_alpha**2*fy**2*(((P1[0] - (Pb[0] + dt*x[6]))*(P1[0] - (Pr[0] + dt*x[3])) + (P1[1] - (Pb[1] + dt*x[7]))*(P1[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1]))) + (P1[2] - (Pc[2] + dt*x[2]))*(P1[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P1[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P1[1] - (Pb[1] + dt*x[7]))))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**4*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)**2*((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*((Pc[2] + dt*x[2])*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2) - (P1[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*(P1[0] - (Pb[0] + dt*x[6])) + (Pc[1] + dt*x[1])*(P1[1] - (Pb[1] + dt*x[7]))) + (P1[2] + (Pb[2] + dt*x[8]))*(P1[0]*(Pb[0] + dt*x[6]) + P1[1]*(Pb[1] + dt*x[7])) - (Pb[2] + dt*x[8])*(P1[0]**2 + P1[1]**2) - P1[2]*((Pb[0] + dt*x[6])**2 + (Pb[1] + dt*x[7])**2))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**6*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_v**2*sigma_bearing**2*fx**2*(((P1[0] - (Pc[0] + dt*x[0]))*(P1[0] - (Pr[0] + dt*x[3])) + (P1[1] - (Pc[1] + dt*x[1]))*(P1[1] - (Pr[1] + dt*x[4])))*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2) + (P1[2] - (Pb[2] + dt*x[8]))*(P1[2] - (Pr[2] + dt*x[5]))*((P1[0] - (Pb[0] + dt*x[6]))*(P1[0] - (Pc[0] + dt*x[0])) + (P1[1] - (Pb[1] + dt*x[7]))*(P1[1] - (Pc[1] + dt*x[1]))))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**4*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2*((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_ranging**2*fy**2*((Pb[2] + dt*x[8])*(P1[0]*cos(thetac + dt*x[9]) + P1[1]*sin(thetac + dt*x[9])) - (Pc[2] + dt*x[2])*(cos(thetac + dt*x[9])*(P1[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P1[1] - (Pb[1] + dt*x[7]))) - P1[2]*((Pb[0] + dt*x[6])*cos(thetac + dt*x[9]) + (Pb[1] + dt*x[7])*sin(thetac + dt*x[9])) + (P1[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*cos(thetac + dt*x[9]) + (Pc[1] + dt*x[1])*sin(thetac + dt*x[9])))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**4*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_u**2*sigma_v**2*(P1[0]*(Pb[0] + dt*x[6]) + (Pr[0] + dt*x[3])*(P1[0] - (Pb[0] + dt*x[6])) + P1[1]*(Pb[1] + dt*x[7]) + (Pr[1] + dt*x[4])*(P1[1] - (Pb[1] + dt*x[7])) + P1[2]*(Pb[2] + dt*x[8]) + (Pr[2] + dt*x[5])*(P1[2] - (Pb[2] + dt*x[8])) - P1[0]**2 - P1[1]**2 - P1[2]**2)**2/(((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2*((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_bearing**2*fy**2*((P1[2] - (Pb[2] + dt*x[8]))*((P1[0] - (Pr[0] + dt*x[3]))*(P1[1] - (Pb[1] + dt*x[7])) + (P1[0] - (Pb[0] + dt*x[6]))*(P1[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1]))) + (P1[2] - (Pc[2] + dt*x[2]))*(cos(thetac + dt*x[9])*(P1[1] - (Pr[1] + dt*x[4])) + sin(thetac + dt*x[9])*(P1[0] - (Pr[0] + dt*x[3])))*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2) + (P1[2] - (Pb[2] + dt*x[8]))*(P1[2] - (Pc[2] + dt*x[2]))*(P1[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P1[1] - (Pb[1] + dt*x[7])) + sin(thetac + dt*x[9])*(P1[0] - (Pb[0] + dt*x[6]))))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**4*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2*((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2)) + \
                         fx**2*((P1[0] - (Pb[0] + dt*x[6]))*(P1[1] - (Pc[1] + dt*x[1])) + (P1[0] - (Pc[0] + dt*x[0]))*(P1[1] - (Pb[1] + dt*x[7])))**2/((cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**4*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)**2)*(sigma_v**2*sigma_alpha**2*(P1[2] - (Pr[2] + dt*x[5]))**2/((P1[0] - (Pr[0] + dt*x[3]))**2 + (P1[1] - (Pr[1] + dt*x[4]))**2 + (P1[2] - (Pr[2] + dt*x[5]))**2) + sigma_ranging**2*sigma_alpha**2*fy**2/(cos(thetac + dt*x[9])*(P1[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P1[1] - (Pc[1] + dt*x[1])))**2 + sigma_v**2*sigma_ranging**2*((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2)/((P1[0] - (Pb[0] + dt*x[6]))**2 + (P1[1] - (Pb[1] + dt*x[7]))**2 + (P1[2] - (Pb[2] + dt*x[8]))**2)**2) \
			 )/(sigma_u**2*sigma_v**2*sigma_ranging**2*sigma_bearing**2*sigma_alpha**2)

		f2 = ( \
                         sigma_bearing**2*sigma_alpha**2*fx**2*fy**2*(P2[0]*(Pr[0] + dt*x[3]) + (Pc[0] + dt*x[0])*(P2[0] - (Pr[0] + dt*x[3])) + P2[1]*(Pr[1] + dt*x[4]) + (Pc[1] + dt*x[1])*(P2[1] - (Pr[1] + dt*x[4])) + P2[2]*(Pr[2] + dt*x[5]) + (Pc[2] + dt*x[2])*(P2[2] - (Pr[2] + dt*x[5])) - P2[0]**2 - P2[1]**2 - P2[2]**2)**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**6*((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_alpha**2*fy**2*(((P2[0] - (Pb[0] + dt*x[6]))*(P2[0] - (Pr[0] + dt*x[3])) + (P2[1] - (Pb[1] + dt*x[7]))*(P2[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1]))) + (P2[2] - (Pc[2] + dt*x[2]))*(P2[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P2[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P2[1] - (Pb[1] + dt*x[7]))))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**4*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)**2*((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*((Pc[2] + dt*x[2])*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2) - (P2[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*(P2[0] - (Pb[0] + dt*x[6])) + (Pc[1] + dt*x[1])*(P2[1] - (Pb[1] + dt*x[7]))) + (P2[2] + (Pb[2] + dt*x[8]))*(P2[0]*(Pb[0] + dt*x[6]) + P2[1]*(Pb[1] + dt*x[7])) - (Pb[2] + dt*x[8])*(P2[0]**2 + P2[1]**2) - P2[2]*((Pb[0] + dt*x[6])**2 + (Pb[1] + dt*x[7])**2))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**6*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_v**2*sigma_bearing**2*fx**2*(((P2[0] - (Pc[0] + dt*x[0]))*(P2[0] - (Pr[0] + dt*x[3])) + (P2[1] - (Pc[1] + dt*x[1]))*(P2[1] - (Pr[1] + dt*x[4])))*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2) + (P2[2] - (Pb[2] + dt*x[8]))*(P2[2] - (Pr[2] + dt*x[5]))*((P2[0] - (Pb[0] + dt*x[6]))*(P2[0] - (Pc[0] + dt*x[0])) + (P2[1] - (Pb[1] + dt*x[7]))*(P2[1] - (Pc[1] + dt*x[1]))))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**4*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2*((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_ranging**2*fy**2*((Pb[2] + dt*x[8])*(P2[0]*cos(thetac + dt*x[9]) + P2[1]*sin(thetac + dt*x[9])) - (Pc[2] + dt*x[2])*(cos(thetac + dt*x[9])*(P2[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P2[1] - (Pb[1] + dt*x[7]))) - P2[2]*((Pb[0] + dt*x[6])*cos(thetac + dt*x[9]) + (Pb[1] + dt*x[7])*sin(thetac + dt*x[9])) + (P2[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*cos(thetac + dt*x[9]) + (Pc[1] + dt*x[1])*sin(thetac + dt*x[9])))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**4*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_u**2*sigma_v**2*(P2[0]*(Pb[0] + dt*x[6]) + (Pr[0] + dt*x[3])*(P2[0] - (Pb[0] + dt*x[6])) + P2[1]*(Pb[1] + dt*x[7]) + (Pr[1] + dt*x[4])*(P2[1] - (Pb[1] + dt*x[7])) + P2[2]*(Pb[2] + dt*x[8]) + (Pr[2] + dt*x[5])*(P2[2] - (Pb[2] + dt*x[8])) - P2[0]**2 - P2[1]**2 - P2[2]**2)**2/(((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2*((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2)) + \
                        sigma_u**2*sigma_bearing**2*fy**2*((P2[2] - (Pb[2] + dt*x[8]))*((P2[0] - (Pr[0] + dt*x[3]))*(P2[1] - (Pb[1] + dt*x[7])) + (P2[0] - (Pb[0] + dt*x[6]))*(P2[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1]))) + (P2[2] - (Pc[2] + dt*x[2]))*(cos(thetac + dt*x[9])*(P2[1] - (Pr[1] + dt*x[4])) + sin(thetac + dt*x[9])*(P2[0] - (Pr[0] + dt*x[3])))*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2) + (P2[2] - (Pb[2] + dt*x[8]))*(P2[2] - (Pc[2] + dt*x[2]))*(P2[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P2[1] - (Pb[1] + dt*x[7])) + sin(thetac + dt*x[9])*(P2[0] - (Pb[0] + dt*x[6]))))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**4*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2*((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2)) + \
                         fx**2*((P2[0] - (Pb[0] + dt*x[6]))*(P2[1] - (Pc[1] + dt*x[1])) + (P2[0] - (Pc[0] + dt*x[0]))*(P2[1] - (Pb[1] + dt*x[7])))**2/((cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**4*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)**2)*(sigma_v**2*sigma_alpha**2*(P2[2] - (Pr[2] + dt*x[5]))**2/((P2[0] - (Pr[0] + dt*x[3]))**2 + (P2[1] - (Pr[1] + dt*x[4]))**2 + (P2[2] - (Pr[2] + dt*x[5]))**2) + sigma_ranging**2*sigma_alpha**2*fy**2/(cos(thetac + dt*x[9])*(P2[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P2[1] - (Pc[1] + dt*x[1])))**2 + sigma_v**2*sigma_ranging**2*((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2)/((P2[0] - (Pb[0] + dt*x[6]))**2 + (P2[1] - (Pb[1] + dt*x[7]))**2 + (P2[2] - (Pb[2] + dt*x[8]))**2)**2) \
			 )/(sigma_u**2*sigma_v**2*sigma_ranging**2*sigma_bearing**2*sigma_alpha**2)

		f3 = ( \
                         sigma_bearing**2*sigma_alpha**2*fx**2*fy**2*(P3[0]*(Pr[0] + dt*x[3]) + (Pc[0] + dt*x[0])*(P3[0] - (Pr[0] + dt*x[3])) + P3[1]*(Pr[1] + dt*x[4]) + (Pc[1] + dt*x[1])*(P3[1] - (Pr[1] + dt*x[4])) + P3[2]*(Pr[2] + dt*x[5]) + (Pc[2] + dt*x[2])*(P3[2] - (Pr[2] + dt*x[5])) - P3[0]**2 - P3[1]**2 - P3[2]**2)**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**6*((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_alpha**2*fy**2*(((P3[0] - (Pb[0] + dt*x[6]))*(P3[0] - (Pr[0] + dt*x[3])) + (P3[1] - (Pb[1] + dt*x[7]))*(P3[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1]))) + (P3[2] - (Pc[2] + dt*x[2]))*(P3[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P3[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P3[1] - (Pb[1] + dt*x[7]))))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**4*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)**2*((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*((Pc[2] + dt*x[2])*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2) - (P3[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*(P3[0] - (Pb[0] + dt*x[6])) + (Pc[1] + dt*x[1])*(P3[1] - (Pb[1] + dt*x[7]))) + (P3[2] + (Pb[2] + dt*x[8]))*(P3[0]*(Pb[0] + dt*x[6]) + P3[1]*(Pb[1] + dt*x[7])) - (Pb[2] + dt*x[8])*(P3[0]**2 + P3[1]**2) - P3[2]*((Pb[0] + dt*x[6])**2 + (Pb[1] + dt*x[7])**2))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**6*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_v**2*sigma_bearing**2*fx**2*(((P3[0] - (Pc[0] + dt*x[0]))*(P3[0] - (Pr[0] + dt*x[3])) + (P3[1] - (Pc[1] + dt*x[1]))*(P3[1] - (Pr[1] + dt*x[4])))*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2) + (P3[2] - (Pb[2] + dt*x[8]))*(P3[2] - (Pr[2] + dt*x[5]))*((P3[0] - (Pb[0] + dt*x[6]))*(P3[0] - (Pc[0] + dt*x[0])) + (P3[1] - (Pb[1] + dt*x[7]))*(P3[1] - (Pc[1] + dt*x[1]))))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**4*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2*((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_ranging**2*fy**2*((Pb[2] + dt*x[8])*(P3[0]*cos(thetac + dt*x[9]) + P3[1]*sin(thetac + dt*x[9])) - (Pc[2] + dt*x[2])*(cos(thetac + dt*x[9])*(P3[0] - (Pb[0] + dt*x[6])) + sin(thetac + dt*x[9])*(P3[1] - (Pb[1] + dt*x[7]))) - P3[2]*((Pb[0] + dt*x[6])*cos(thetac + dt*x[9]) + (Pb[1] + dt*x[7])*sin(thetac + dt*x[9])) + (P3[2] - (Pb[2] + dt*x[8]))*((Pc[0] + dt*x[0])*cos(thetac + dt*x[9]) + (Pc[1] + dt*x[1])*sin(thetac + dt*x[9])))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**4*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2) + \
                         sigma_u**2*sigma_v**2*(P3[0]*(Pb[0] + dt*x[6]) + (Pr[0] + dt*x[3])*(P3[0] - (Pb[0] + dt*x[6])) + P3[1]*(Pb[1] + dt*x[7]) + (Pr[1] + dt*x[4])*(P3[1] - (Pb[1] + dt*x[7])) + P3[2]*(Pb[2] + dt*x[8]) + (Pr[2] + dt*x[5])*(P3[2] - (Pb[2] + dt*x[8])) - P3[0]**2 - P3[1]**2 - P3[2]**2)**2/(((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2*((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2)) + \
                         sigma_u**2*sigma_bearing**2*fy**2*((P3[2] - (Pb[2] + dt*x[8]))*((P3[0] - (Pr[0] + dt*x[3]))*(P3[1] - (Pb[1] + dt*x[7])) + (P3[0] - (Pb[0] + dt*x[6]))*(P3[1] - (Pr[1] + dt*x[4])))*(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1]))) + (P3[2] - (Pc[2] + dt*x[2]))*(cos(thetac + dt*x[9])*(P3[1] - (Pr[1] + dt*x[4])) + sin(thetac + dt*x[9])*(P3[0] - (Pr[0] + dt*x[3])))*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2) + (P3[2] - (Pb[2] + dt*x[8]))*(P3[2] - (Pc[2] + dt*x[2]))*(P3[2] - (Pr[2] + dt*x[5]))*(cos(thetac + dt*x[9])*(P3[1] - (Pb[1] + dt*x[7])) + sin(thetac + dt*x[9])*(P3[0] - (Pb[0] + dt*x[6]))))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**4*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2*((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2)) + \
                         fx**2*((P3[0] - (Pb[0] + dt*x[6]))*(P3[1] - (Pc[1] + dt*x[1])) + (P3[0] - (Pc[0] + dt*x[0]))*(P3[1] - (Pb[1] + dt*x[7])))**2/((cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**4*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)**2)*(sigma_v**2*sigma_alpha**2*(P3[2] - (Pr[2] + dt*x[5]))**2/((P3[0] - (Pr[0] + dt*x[3]))**2 + (P3[1] - (Pr[1] + dt*x[4]))**2 + (P3[2] - (Pr[2] + dt*x[5]))**2) + sigma_ranging**2*sigma_alpha**2*fy**2/(cos(thetac + dt*x[9])*(P3[0] - (Pc[0] + dt*x[0])) + sin(thetac + dt*x[9])*(P3[1] - (Pc[1] + dt*x[1])))**2 + sigma_v**2*sigma_ranging**2*((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2)/((P3[0] - (Pb[0] + dt*x[6]))**2 + (P3[1] - (Pb[1] + dt*x[7]))**2 + (P3[2] - (Pb[2] + dt*x[8]))**2)**2) \
			 )/(sigma_u**2*sigma_v**2*sigma_ranging**2*sigma_bearing**2*sigma_alpha**2)

		out["F"] = [1/f1, 1/f2, 1/f3]
		out["G"] = (A.dot(x.reshape((10,1))) - b).reshape(42,).tolist()

def odom(msg):
	global P1,P2,P3,Pc,Pr,Pb,A,b,thetac
	
	Pc = np.array(msg.data[18:21])
	Pr = np.array(msg.data[21:24])
	Pb = np.array(msg.data[24:27])
	P1 = np.array(msg.data[0:3])
	P2 = np.array(msg.data[6:9])
	P3 = np.array(msg.data[12:15])
	thetac = msg.data[27]
	
	nc = np.array([cos(thetac),sin(thetac),0])
	nc_dot = np.array([-sin(thetac),cos(thetac),0])
	r1c = np.array([P1[0] - Pc[0],P1[1] - Pc[1],P1[2] - Pc[2]])
	r2c = np.array([P2[0] - Pc[0],P2[1] - Pc[1],P2[2] - Pc[2]])
	r3c = np.array([P3[0] - Pc[0],P3[1] - Pc[1],P3[2] - Pc[2]])

	A = np.array([ \
				  (-2*(Pc-P1)[:2]).tolist()+[0]*8, \
				  (-2*(Pc-P2)[:2]).tolist()+[0]*8, \
				  (-2*(Pc-P3)[:2]).tolist()+[0]*8, \
				  [0]*3+(-2*(Pr-P1)[:2]).tolist()+[0]*5, \
				  [0]*3+(-2*(Pr-P2)[:2]).tolist()+[0]*5, \
				  [0]*3+(-2*(Pr-P3)[:2]).tolist()+[0]*5, \
				  [0]*6+(-2*(Pb-P1)[:2]).tolist()+[0]*2, \
				  [0]*6+(-2*(Pb-P2)[:2]).tolist()+[0]*2, \
				  [0]*6+(-2*(Pb-P3)[:2]).tolist()+[0]*2, \
				  (2*(Pc-P1)[:2]).tolist()+[0]*8, \
				  (2*(Pc-P2)[:2]).tolist()+[0]*8, \
				  (2*(Pc-P3)[:2]).tolist()+[0]*8, \
				  [0]*3+(2*(Pr-P1)[:2]).tolist()+[0]*5, \
				  [0]*3+(2*(Pr-P2)[:2]).tolist()+[0]*5, \
				  [0]*3+(2*(Pr-P3)[:2]).tolist()+[0]*5, \
				  [0]*6+(2*(Pb-P1)[:2]).tolist()+[0]*2, \
				  [0]*6+(2*(Pb-P2)[:2]).tolist()+[0]*2, \
				  [0]*6+(2*(Pb-P3)[:2]).tolist()+[0]*2, \
				  (-2*(Pc-Pr)[:2]).tolist()+[0]*8, \
				  (-2*(Pc-Pb)[:2]).tolist()+[0]*8, \
				  [0]*3+(-2*(Pr-Pc)[:2]).tolist()+[0]*5, \
				  [0]*3+(-2*(Pr-Pb)[:2]).tolist()+[0]*5, \
				  [0]*6+(-2*(Pb-Pc)[:2]).tolist()+[0]*2, \
				  [0]*6+(-2*(Pb-Pr)[:2]).tolist()+[0]*2, \
				  (2*(Pc-Pr)[:2]).tolist()+[0]*8, \
				  (2*(Pc-Pb)[:2]).tolist()+[0]*8, \
				  [0]*3+(2*(Pr-Pc)[:2]).tolist()+[0]*5, \
				  [0]*3+(2*(Pr-Pb)[:2]).tolist()+[0]*5, \
				  [0]*6+(2*(Pb-Pc)[:2]).tolist()+[0]*2, \
				  [0]*6+(2*(Pb-Pr)[:2]).tolist()+[0]*2, \
				  np.concatenate((-(nc.dot(r1c)*r1c[:2]/np.linalg.norm(r1c[:2])**3-nc[:2]/np.linalg.norm(r1c[:2]))/sqrt(1 - nc.dot(r1c)**2/np.linalg.norm(r1c[:2])**2),[0]*7,[-nc_dot.dot(r1c)/np.linalg.norm(r1c[:2])/sqrt(1 - nc.dot(r1c)**2/np.linalg.norm(r1c[:2])**2)])), \
				  np.concatenate((-(nc.dot(r2c)*r2c[:2]/np.linalg.norm(r2c[:2])**3-nc[:2]/np.linalg.norm(r2c[:2]))/sqrt(1 - nc.dot(r2c)**2/np.linalg.norm(r2c[:2])**2),[0]*7,[-nc_dot.dot(r2c)/np.linalg.norm(r2c[:2])/sqrt(1 - nc.dot(r2c)**2/np.linalg.norm(r2c[:2])**2)])), \
				  np.concatenate((-(nc.dot(r3c)*r3c[:2]/np.linalg.norm(r3c[:2])**3-nc[:2]/np.linalg.norm(r3c[:2]))/sqrt(1 - nc.dot(r3c)**2/np.linalg.norm(r3c[:2])**2),[0]*7,[-nc_dot.dot(r3c)/np.linalg.norm(r3c[:2])/sqrt(1 - nc.dot(r3c)**2/np.linalg.norm(r3c[:2])**2)])), \
				  np.concatenate((abs(r1c[2])*nc[:2]/nc.dot(r1c)**2/(1 + r1c[2]**2/nc.dot(r1c)**2),[-r1c[2]/nc.dot(r1c)/abs(r1c[2])/(1 + r1c[2]**2/nc.dot(r1c)**2)],[0]*6,[-abs(r1c[2])*nc_dot.dot(r1c)/nc.dot(r1c)**2/(1 + r1c[2]**2/nc.dot(r1c)**2)])), \
				  np.concatenate((abs(r2c[2])*nc[:2]/nc.dot(r2c)**2/(1 + r2c[2]**2/nc.dot(r2c)**2),[-r2c[2]/nc.dot(r2c)/abs(r2c[2])/(1 + r2c[2]**2/nc.dot(r2c)**2)],[0]*6,[-abs(r2c[2])*nc_dot.dot(r2c)/nc.dot(r2c)**2/(1 + r2c[2]**2/nc.dot(r2c)**2)])), \
				  np.concatenate((abs(r3c[2])*nc[:2]/nc.dot(r3c)**2/(1 + r3c[2]**2/nc.dot(r3c)**2),[-r3c[2]/nc.dot(r3c)/abs(r3c[2])/(1 + r3c[2]**2/nc.dot(r3c)**2)],[0]*6,[-abs(r3c[2])*nc_dot.dot(r3c)/nc.dot(r3c)**2/(1 + r3c[2]**2/nc.dot(r3c)**2)])), \
				  [0]*2+[-1]+[0]*7, \
				  [0]*5+[-1]+[0]*4, \
				  [0]*8+[-1]+[0], \
				  [0]*2+[1]+[0]*7, \
				  [0]*5+[1]+[0]*4, \
				  [0]*8+[1]+[0] \
				  ])
	
	b = np.array([ \
				  [np.linalg.norm((Pc-P1)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pc-P2)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pc-P3)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pr-P1)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pr-P2)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pr-P3)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pb-P1)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pb-P2)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pb-P3)[:2])**2 - d_safe_car**2], \
				  [d_measuring**2 - np.linalg.norm((Pc-P1)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pc-P2)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pc-P3)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pr-P1)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pr-P2)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pr-P3)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P1)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P2)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P3)[:2])**2], \
				  [np.linalg.norm((Pc-Pr)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pc-Pb)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pr-Pc)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pr-Pb)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pb-Pc)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pb-Pr)[:2])**2 - d_safe_uav**2], \
				  [d_communication**2 - np.linalg.norm((Pc-Pr)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pc-Pb)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pr-Pc)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pr-Pb)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pb-Pc)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pb-Pr)[:2])**2], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(nc.dot(r1c)/np.linalg.norm(r1c[:2]))], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(nc.dot(r2c)/np.linalg.norm(r2c[:2]))], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(nc.dot(r3c)/np.linalg.norm(r3c[:2]))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(abs(r1c[2]),nc.dot(r1c))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(abs(r2c[2]),nc.dot(r2c))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(abs(r3c[2]),nc.dot(r3c))], \
				  [Pc[2] - height_l], \
				  [Pr[2] - height_l], \
				  [Pb[2] - height_l], \
				  [height_u - Pc[2]], \
				  [height_u - Pr[2]], \
				  [height_u - Pb[2]] \
				  ])
	
def	qpsolver():
	global camera_cmd_vel,ranging_cmd_vel,bearing_cmd_vel,time_last,dt
	
	objective = Objective()
	algorithm = NSGA2(pop_size=20,n_offsprings=None,sampling=get_sampling("real_random"), \
					  crossover=get_crossover("real_sbx", prob=0.9, eta=15), \
					  mutation=get_mutation("real_pm", eta=20), eliminate_duplicates=True)
	termination = get_termination("n_gen", 8)
	dt = rospy.Time.now().to_sec() - time_last
	res = minimize(objective, algorithm, termination, seed=10, save_history=False, verbose=True, return_least_infeasible=True)
	
	tmp = np.cumsum(res.F,axis=1)
	num_opt = np.argmin(tmp[:,2])
	
	camera_cmd_vel.linear.x = res.X[num_opt,0]
	camera_cmd_vel.linear.y = res.X[num_opt,1]
	camera_cmd_vel.linear.z = res.X[num_opt,2]
	ranging_cmd_vel.linear.x = res.X[num_opt,3]
	ranging_cmd_vel.linear.y = res.X[num_opt,4]
	ranging_cmd_vel.linear.z = res.X[num_opt,5]
	bearing_cmd_vel.linear.x = res.X[num_opt,6]
	bearing_cmd_vel.linear.y = res.X[num_opt,7]
	bearing_cmd_vel.linear.z = res.X[num_opt,8]
	camera_cmd_vel.angular.z = res.X[num_opt,9]
	
	px4_camera.vel_control(camera_cmd_vel)
	px4_ranging.vel_control(ranging_cmd_vel)
	px4_bearing.vel_control(bearing_cmd_vel)
	time_last = rospy.Time.now().to_sec()

if __name__ == '__main__':
	try:
		rospy.init_node('controller')
		uavtype = ["iris_camera","iris_ranging","iris_bearing"]
		px4_camera = Px4Controller(uavtype[0])
		px4_ranging = Px4Controller(uavtype[1])
		px4_bearing = Px4Controller(uavtype[2])
		rospy.Subscriber('/state', Float64MultiArray, odom, queue_size=10)
		rate = rospy.Rate(100)
		while b is None:
			rate.sleep()

		while not rospy.is_shutdown():
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
