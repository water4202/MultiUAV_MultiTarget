#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
import threading
from gazebo_msgs.msg import ModelStates
from scipy.optimize import minimize
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from std_msgs.msg import Float64MultiArray

P1,P2,P3,Pc,Pr,Pb,thetac,A,b = None,None,None,None,None,None,None,None,None
camera_cmd_vel = Twist()
ranging_cmd_vel = Twist()
bearing_cmd_vel = Twist()
fx,fy,lx,ly = 0.1496485702,0.1496485702,0.1693333333,0.127
#fx,fy,lx,ly = 565.6,565.6,640,480
#sigma_u,sigma_v,sigma_ranging,sigma_bearing = 0.007,0.007,0.01,0.01
sigma_u,sigma_v,sigma_ranging,sigma_bearing = 1,1,1,1
x_fov_wealth = 3*pi/180
y_fov_wealth = 3*pi/180
height_l = 0.3
height_u = 100
d_safe_car = 0.7
d_measuring = 2.2
d_safe_uav = 0.7
d_communication = 20
gamma = 1.0

def object_fun(x):
        return ( sigma_u**2*sigma_v**2*((P1[0] - (Pb[0] + x[6]))*(P1[0] - (Pr[0] + x[3])) + (P1[1] - (Pb[1] + x[7]))*(P1[1] - (Pr[1] + x[4])))**2/(((P1[0] - (Pb[0] + x[6]))**2 + (P1[1] - (Pb[1] + x[7]))**2)**2*((P1[0] - (Pr[0] + x[3]))**2 + (P1[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_bearing**2*(sigma_v**2*fx**2*((P1[0] - (Pr[0] + x[3]))*(P1[0] - (Pc[0] + x[0])) + (P1[1] - (Pr[1] + x[4]))*(P1[1] - (Pc[1] + x[1])))**2 + sigma_u**2*fy**2*(P1[2] - (Pc[2] + x[2]))**2*((P1[1] - (Pr[1] + x[4]))*cos(thetac + x[9]) - (P1[0] - (Pr[0] + x[3]))*sin(thetac + x[9]))**2)/((cos(thetac + x[9])*(P1[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P1[1] - (Pc[1] + x[1])))**4*((P1[0] - (Pr[0] + x[3]))**2 + (P1[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_ranging**2*(sigma_u**2*fx**2*((P1[0] - (Pb[0] + x[6]))*(P1[1] - (Pc[1] + x[1])) - (P1[1] - (Pb[1] + x[7]))*(P1[0] - (Pc[0] + x[0])))**2 + sigma_v**2*fy**2*(P1[2] - (Pc[2] + x[2]))**2*((P1[1] - (Pb[1] + x[7]))*sin(thetac + x[9]) + (P1[0] - (Pb[0] + x[6]))*cos(thetac + x[9]))**2)/((cos(thetac + x[9])*(P1[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P1[1] - (Pc[1] + x[1])))**4*((P1[0] - (Pb[0] + x[6]))**2 + (P1[1] - (Pb[1] + x[7]))**2)**2) \
                  + sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*(P1[2] - (Pc[2] + x[2]))**2/(cos(thetac + x[9])*(P1[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P1[1] - (Pc[1] + x[1])))**6 \
                )*( \
                  sigma_u**2*sigma_v**2*((P2[0] - (Pb[0] + x[6]))*(P2[0] - (Pr[0] + x[3])) + (P2[1] - (Pb[1] + x[7]))*(P2[1] - (Pr[1] + x[4])))**2/(((P2[0] - (Pb[0] + x[6]))**2 + (P2[1] - (Pb[1] + x[7]))**2)**2*((P2[0] - (Pr[0] + x[3]))**2 + (P2[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_bearing**2*(sigma_v**2*fx**2*((P2[0] - (Pr[0] + x[3]))*(P2[0] - (Pc[0] + x[0])) + (P2[1] - (Pr[1] + x[4]))*(P2[1] - (Pc[1] + x[1])))**2 + sigma_u**2*fy**2*(P2[2] - (Pc[2] + x[2]))**2*((P2[1] - (Pr[1] + x[4]))*cos(thetac + x[9]) - (P2[0] - (Pr[0] + x[3]))*sin(thetac + x[9]))**2)/((cos(thetac + x[9])*(P2[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P2[1] - (Pc[1] + x[1])))**4*((P2[0] - (Pr[0] + x[3]))**2 + (P2[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_ranging**2*(sigma_u**2*fx**2*((P2[0] - (Pb[0] + x[6]))*(P2[1] - (Pc[1] + x[1])) - (P2[1] - (Pb[1] + x[7]))*(P2[0] - (Pc[0] + x[0])))**2 + sigma_v**2*fy**2*(P2[2] - (Pc[2] + x[2]))**2*((P2[1] - (Pb[1] + x[7]))*sin(thetac + x[9]) + (P2[0] - (Pb[0] + x[6]))*cos(thetac + x[9]))**2)/((cos(thetac + x[9])*(P2[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P2[1] - (Pc[1] + x[1])))**4*((P2[0] - (Pb[0] + x[6]))**2 + (P2[1] - (Pb[1] + x[7]))**2)**2) \
                  + sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*(P2[2] - (Pc[2] + x[2]))**2/(cos(thetac + x[9])*(P2[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P2[1] - (Pc[1] + x[1])))**6 \
                )*( \
                  sigma_u**2*sigma_v**2*((P3[0] - (Pb[0] + x[6]))*(P3[0] - (Pr[0] + x[3])) + (P3[1] - (Pb[1] + x[7]))*(P3[1] - (Pr[1] + x[4])))**2/(((P3[0] - (Pb[0] + x[6]))**2 + (P3[1] - (Pb[1] + x[7]))**2)**2*((P3[0] - (Pr[0] + x[3]))**2 + (P3[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_bearing**2*(sigma_v**2*fx**2*((P3[0] - (Pr[0] + x[3]))*(P3[0] - (Pc[0] + x[0])) + (P3[1] - (Pr[1] + x[4]))*(P3[1] - (Pc[1] + x[1])))**2 + sigma_u**2*fy**2*(P3[2] - (Pc[2] + x[2]))**2*((P3[1] - (Pr[1] + x[4]))*cos(thetac + x[9]) - (P3[0] - (Pr[0] + x[3]))*sin(thetac + x[9]))**2)/((cos(thetac + x[9])*(P3[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P3[1] - (Pc[1] + x[1])))**4*((P3[0] - (Pr[0] + x[3]))**2 + (P3[1] - (Pr[1] + x[4]))**2)) \
                  + sigma_ranging**2*(sigma_u**2*fx**2*((P3[0] - (Pb[0] + x[6]))*(P3[1] - (Pc[1] + x[1])) - (P3[1] - (Pb[1] + x[7]))*(P3[0] - (Pc[0] + x[0])))**2 + sigma_v**2*fy**2*(P3[2] - (Pc[2] + x[2]))**2*((P3[1] - (Pb[1] + x[7]))*sin(thetac + x[9]) + (P3[0] - (Pb[0] + x[6]))*cos(thetac + x[9]))**2)/((cos(thetac + x[9])*(P3[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P3[1] - (Pc[1] + x[1])))**4*((P3[0] - (Pb[0] + x[6]))**2 + (P3[1] - (Pb[1] + x[7]))**2)**2) \
                  + sigma_ranging**2*sigma_bearing**2*fx**2*fy**2*(P3[2] - (Pc[2] + x[2]))**2/(cos(thetac + x[9])*(P3[0] - (Pc[0] + x[0])) + sin(thetac + x[9])*(P3[1] - (Pc[1] + x[1])))**6 \
                )/(sigma_u**6*sigma_v**6*sigma_ranging**6*sigma_bearing**6)

def cons_maker(i=0):
	def constraint(x):
		return b[i] - A[i,0]*x[0] - A[i,1]*x[1] - A[i,2]*x[2] - A[i,3]*x[3] - A[i,4]*x[4] - A[i,5]*x[5] - A[i,6]*x[6] - A[i,7]*x[7] - A[i,8]*x[8] - A[i,9]*x[9] - x[i+10]
	return constraint

def cons_maker1(i=0):
	def constraint(x):
		return x[i+10]
	return constraint

def odom(msg):
	global P1,P2,P3,Pc,Pr,Pb,A,b,thetac
	'''
	Pc = np.array([msg.data[18], msg.data[19], msg.data[20]])
	Pr = np.array([msg.data[21], msg.data[22], msg.data[23]])
	Pb = np.array([msg.data[24], msg.data[25], msg.data[26]])
	P1 = np.array([msg.data[0], msg.data[1], msg.data[2]])
	P2 = np.array([msg.data[6], msg.data[7], msg.data[8]])
	P3 = np.array([msg.data[12], msg.data[13], msg.data[14]])
	thetac = msg.data[27]
	'''
	UAV1_index = msg.name.index('iris_camera')
	UAV2_index = msg.name.index('iris_ranging')
	UAV3_index = msg.name.index('iris_bearing')
	car1_index = msg.name.index('car1')
	car2_index = msg.name.index('car2')
	car3_index = msg.name.index('car3')
	Pc = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	Pr = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
	Pb = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
	P1 = np.array([msg.pose[car1_index].position.x, msg.pose[car1_index].position.y, msg.pose[car1_index].position.z])
	P2 = np.array([msg.pose[car2_index].position.x, msg.pose[car2_index].position.y, msg.pose[car2_index].position.z])
	P3 = np.array([msg.pose[car3_index].position.x, msg.pose[car3_index].position.y, msg.pose[car3_index].position.z])
	
	nc = np.array([cos(thetac),sin(thetac),0])
	nc_dot = np.array([-sin(thetac),cos(thetac),0])
	r1c_xy = np.array([P1[0] - Pc[0],P1[1] - Pc[1],0])
	r2c_xy = np.array([P2[0] - Pc[0],P2[1] - Pc[1],0])
	r3c_xy = np.array([P3[0] - Pc[0],P3[1] - Pc[1],0])
	r1c_z = np.array([0,0,P1[2] - Pc[2]])
	r2c_z = np.array([0,0,P2[2] - Pc[2]])
	r3c_z = np.array([0,0,P3[2] - Pc[2]])

	A = np.array([[-2*(Pc[0]-P1[0]), -2*(Pc[1]-P1[1])]+[0]*8, \
				  [-2*(Pc[0]-P2[0]), -2*(Pc[1]-P2[1])]+[0]*8, \
				  [-2*(Pc[0]-P3[0]), -2*(Pc[1]-P3[1])]+[0]*8, \
				  [0]*3+[-2*(Pr[0]-P1[0]), -2*(Pr[1]-P1[1])]+[0]*5, \
				  [0]*3+[-2*(Pr[0]-P2[0]), -2*(Pr[1]-P2[1])]+[0]*5, \
				  [0]*3+[-2*(Pr[0]-P3[0]), -2*(Pr[1]-P3[1])]+[0]*5, \
				  [0]*6+[-2*(Pb[0]-P1[0]), -2*(Pb[1]-P1[1])]+[0]*2, \
				  [0]*6+[-2*(Pb[0]-P2[0]), -2*(Pb[1]-P2[1])]+[0]*2, \
				  [0]*6+[-2*(Pb[0]-P3[0]), -2*(Pb[1]-P3[1])]+[0]*2, \
				  [2*(Pc[0]-P1[0]), 2*(Pc[1]-P1[1])]+[0]*8, \
				  [2*(Pc[0]-P2[0]), 2*(Pc[1]-P2[1])]+[0]*8, \
				  [2*(Pc[0]-P3[0]), 2*(Pc[1]-P3[1])]+[0]*8, \
				  [0]*3+[2*(Pr[0]-P1[0]), 2*(Pr[1]-P1[1])]+[0]*5, \
				  [0]*3+[2*(Pr[0]-P2[0]), 2*(Pr[1]-P2[1])]+[0]*5, \
				  [0]*3+[2*(Pr[0]-P3[0]), 2*(Pr[1]-P3[1])]+[0]*5, \
				  [0]*6+[2*(Pb[0]-P1[0]), 2*(Pb[1]-P1[1])]+[0]*2, \
				  [0]*6+[2*(Pb[0]-P2[0]), 2*(Pb[1]-P2[1])]+[0]*2, \
				  [0]*6+[2*(Pb[0]-P3[0]), 2*(Pb[1]-P3[1])]+[0]*2, \
				  [-2*(Pc[0]-Pr[0]), -2*(Pc[1]-Pr[1])]+[0]*8, \
				  [-2*(Pc[0]-Pb[0]), -2*(Pc[1]-Pb[1])]+[0]*8, \
				  [0]*3+[-2*(Pr[0]-Pc[0]), -2*(Pr[1]-Pc[1])]+[0]*5, \
				  [0]*3+[-2*(Pr[0]-Pb[0]), -2*(Pr[1]-Pb[1])]+[0]*5, \
				  [0]*6+[-2*(Pb[0]-Pc[0]), -2*(Pb[1]-Pc[1])]+[0]*2, \
				  [0]*6+[-2*(Pb[0]-Pr[0]), -2*(Pb[1]-Pr[1])]+[0]*2, \
				  [2*(Pc[0]-Pr[0]), 2*(Pc[1]-Pr[1])]+[0]*8, \
				  [2*(Pc[0]-Pb[0]), 2*(Pc[1]-Pb[1])]+[0]*8, \
				  [0]*3+[2*(Pr[0]-Pc[0]), 2*(Pr[1]-Pc[1])]+[0]*5, \
				  [0]*3+[2*(Pr[0]-Pb[0]), 2*(Pr[1]-Pb[1])]+[0]*5, \
				  [0]*6+[2*(Pb[0]-Pc[0]), 2*(Pb[1]-Pc[1])]+[0]*2, \
				  [0]*6+[2*(Pb[0]-Pr[0]), 2*(Pb[1]-Pr[1])]+[0]*2, \
				  np.append(-(np.dot(nc,r1c_xy)*r1c_xy/np.linalg.norm(r1c_xy)**3-nc/np.linalg.norm(r1c_xy))/sqrt(1 - np.dot(nc,r1c_xy)**2/np.linalg.norm(r1c_xy)**2),np.append([0]*6,-np.dot(nc_dot,r1c_xy)/np.linalg.norm(r1c_xy)/sqrt(1 - np.dot(nc,r1c_xy)**2/np.linalg.norm(r1c_xy)**2))), \
				  np.append(-(np.dot(nc,r2c_xy)*r2c_xy/np.linalg.norm(r2c_xy)**3-nc/np.linalg.norm(r2c_xy))/sqrt(1 - np.dot(nc,r2c_xy)**2/np.linalg.norm(r2c_xy)**2),np.append([0]*6,-np.dot(nc_dot,r2c_xy)/np.linalg.norm(r2c_xy)/sqrt(1 - np.dot(nc,r2c_xy)**2/np.linalg.norm(r2c_xy)**2))), \
				  np.append(-(np.dot(nc,r3c_xy)*r3c_xy/np.linalg.norm(r3c_xy)**3-nc/np.linalg.norm(r3c_xy))/sqrt(1 - np.dot(nc,r3c_xy)**2/np.linalg.norm(r3c_xy)**2),np.append([0]*6,-np.dot(nc_dot,r3c_xy)/np.linalg.norm(r3c_xy)/sqrt(1 - np.dot(nc,r3c_xy)**2/np.linalg.norm(r3c_xy)**2))), \
				  np.append((np.linalg.norm(r1c_z)*nc/np.dot(nc,r1c_xy)**2-r1c_z/np.linalg.norm(r1c_z)/np.dot(nc,r1c_xy))/(1 + np.linalg.norm(r1c_z)**2/np.dot(nc,r1c_xy)**2),np.append([0]*6,-np.linalg.norm(r1c_z)*np.dot(nc_dot,r1c_xy)/np.dot(nc,r1c_xy)**2/(1 + np.linalg.norm(r1c_z)**2/np.dot(nc,r1c_xy)**2))), \
				  np.append((np.linalg.norm(r2c_z)*nc/np.dot(nc,r2c_xy)**2-r2c_z/np.linalg.norm(r2c_z)/np.dot(nc,r2c_xy))/(1 + np.linalg.norm(r2c_z)**2/np.dot(nc,r2c_xy)**2),np.append([0]*6,-np.linalg.norm(r2c_z)*np.dot(nc_dot,r2c_xy)/np.dot(nc,r2c_xy)**2/(1 + np.linalg.norm(r2c_z)**2/np.dot(nc,r2c_xy)**2))), \
				  np.append((np.linalg.norm(r3c_z)*nc/np.dot(nc,r3c_xy)**2-r3c_z/np.linalg.norm(r3c_z)/np.dot(nc,r3c_xy))/(1 + np.linalg.norm(r3c_z)**2/np.dot(nc,r3c_xy)**2),np.append([0]*6,-np.linalg.norm(r3c_z)*np.dot(nc_dot,r3c_xy)/np.dot(nc,r3c_xy)**2/(1 + np.linalg.norm(r3c_z)**2/np.dot(nc,r3c_xy)**2))), \
				  [0]*2+[-1]+[0]*7, \
				  [0]*5+[-1]+[0]*4, \
				  [0]*8+[-1]+[0] \
				  ])

	b = np.array([[np.linalg.norm([Pc[0]-P1[0],Pc[1]-P1[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pc[0]-P2[0],Pc[1]-P2[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pc[0]-P3[0],Pc[1]-P3[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pr[0]-P1[0],Pr[1]-P1[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pr[0]-P2[0],Pr[1]-P2[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pr[0]-P3[0],Pr[1]-P3[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pb[0]-P1[0],Pb[1]-P1[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pb[0]-P2[0],Pb[1]-P2[1]])**2 - d_safe_car**2], \
				  [np.linalg.norm([Pb[0]-P3[0],Pb[1]-P3[1]])**2 - d_safe_car**2], \
				  [d_measuring**2 - np.linalg.norm([Pc[0]-P1[0],Pc[1]-P1[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pc[0]-P2[0],Pc[1]-P2[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pc[0]-P3[0],Pc[1]-P3[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pr[0]-P1[0],Pr[1]-P1[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pr[0]-P2[0],Pr[1]-P2[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pr[0]-P3[0],Pr[1]-P3[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pb[0]-P1[0],Pb[1]-P1[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pb[0]-P2[0],Pb[1]-P2[1]])**2], \
				  [d_measuring**2 - np.linalg.norm([Pb[0]-P3[0],Pb[1]-P3[1]])**2], \
				  [np.linalg.norm([Pc[0]-Pr[0],Pc[1]-Pr[1]])**2 - d_safe_uav**2], \
				  [np.linalg.norm([Pc[0]-Pb[0],Pc[1]-Pb[1]])**2 - d_safe_uav**2], \
				  [np.linalg.norm([Pr[0]-Pc[0],Pr[1]-Pc[1]])**2 - d_safe_uav**2], \
				  [np.linalg.norm([Pr[0]-Pb[0],Pr[1]-Pb[1]])**2 - d_safe_uav**2], \
				  [np.linalg.norm([Pb[0]-Pc[0],Pb[1]-Pc[1]])**2 - d_safe_uav**2], \
				  [np.linalg.norm([Pb[0]-Pr[0],Pb[1]-Pr[1]])**2 - d_safe_uav**2], \
				  [d_communication**2 - np.linalg.norm([Pc[0]-Pr[0],Pc[1]-Pr[1]])**2], \
				  [d_communication**2 - np.linalg.norm([Pc[0]-Pb[0],Pc[1]-Pb[1]])**2], \
				  [d_communication**2 - np.linalg.norm([Pr[0]-Pc[0],Pr[1]-Pc[1]])**2], \
				  [d_communication**2 - np.linalg.norm([Pr[0]-Pb[0],Pr[1]-Pb[1]])**2], \
				  [d_communication**2 - np.linalg.norm([Pb[0]-Pc[0],Pb[1]-Pc[1]])**2], \
				  [d_communication**2 - np.linalg.norm([Pb[0]-Pr[0],Pb[1]-Pr[1]])**2], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(np.dot(nc,r1c_xy)/np.linalg.norm(r1c_xy))], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(np.dot(nc,r2c_xy)/np.linalg.norm(r2c_xy))], \
				  [atan2(lx,2*fx) - x_fov_wealth - acos(np.dot(nc,r3c_xy)/np.linalg.norm(r3c_xy))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(np.linalg.norm(r1c_z),np.dot(nc,r1c_xy))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(np.linalg.norm(r2c_z),np.dot(nc,r2c_xy))], \
				  [atan2(ly,2*fy) - y_fov_wealth - atan2(np.linalg.norm(r3c_z),np.dot(nc,r3c_xy))], \
				  [Pc[2] - height_l], \
				  [Pr[2] - height_l], \
				  [Pb[2] - height_l] \
				  ])

def	qpsolver():
	global camera_cmd_vel,ranging_cmd_vel,bearing_cmd_vel
	
	cons = []
	
	for i in range (b.size):
		cons.append({'type': 'eq', 'fun': cons_maker(i)})
	for i in range (b.size):
		cons.append({'type': 'ineq', 'fun': cons_maker1(i)})
	
	ini = tuple(np.zeros(b.size + 10))
	bnds = ((-0.3, 0.3),)*10 + ((0, np.inf),)*b.size
	
	optimal = minimize(object_fun, ini, method='SLSQP', bounds=bnds, constraints=cons,options={'maxiter':1000}).x
	print(object_fun(optimal[:10]))

	camera_cmd_vel.linear.x = optimal[0]
	camera_cmd_vel.linear.y = optimal[1]
	camera_cmd_vel.linear.z = optimal[2]
	ranging_cmd_vel.linear.x = optimal[3]
	ranging_cmd_vel.linear.y = optimal[4]
	ranging_cmd_vel.linear.z = optimal[5]
	bearing_cmd_vel.linear.x = optimal[6]
	bearing_cmd_vel.linear.y = optimal[7]
	bearing_cmd_vel.linear.z = optimal[8]
	camera_cmd_vel.angular.z = optimal[9]
	
	px4_camera.vel_control(camera_cmd_vel)
	px4_ranging.vel_control(ranging_cmd_vel)
	px4_bearing.vel_control(bearing_cmd_vel)
	
if __name__ == '__main__':
	try:
		rospy.init_node('controller')
		uavtype = ["iris_camera","iris_ranging","iris_bearing"]
		px4_camera = Px4Controller(uavtype[0])
		px4_ranging = Px4Controller(uavtype[1])
		px4_bearing = Px4Controller(uavtype[2])
		rate = rospy.Rate(50)
		
		while thetac == None:
			thetac = px4_camera.current_heading

		while not rospy.is_shutdown():
			msg = rospy.wait_for_message('/gazebo/model_states', ModelStates)
			thetac = px4_camera.current_heading	
			'''
			msg = rospy.wait_for_message('/state', Float64MultiArray)			
			'''			
			odom(msg)
		
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
