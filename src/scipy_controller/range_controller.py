#!/usr/bin/python

import rospy
from math import sin,cos,sqrt,atan2,acos,pi
import numpy as np
import threading
from scipy.optimize import minimize
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from std_msgs.msg import Float64MultiArray

P1,P2,P3,Pc,Pr,Pb,A,b = None,None,None,None,None,None,None,None
range_cmd_vel = Twist()
sigma_u,sigma_v,sigma_range,sigma_bearing,sigma_alpha = 0.007,0.007,0.01,0.01,0.01
height_l = 0.3
height_u = 1.8
d_safe_car = 0.7
d_measuring = 2.2
d_safe_uav = 0.7
d_communication = 20

def object_fun(x):
	Pcen = (P1+P2+P3)/3
	return 1/((Pcen[0] - Pb[0] - x[0])**2 + (Pcen[1] - Pb[1] - x[1])**2 + (Pcen[2] - Pb[2] - x[2])**2)

def cons_maker(i):
	def constraint(x):
		return b[i] - A[i,0]*x[0] - A[i,1]*x[1] - A[i,2]*x[2] - x[i+3]
	return constraint

def slack(i):
	def constraint(x):
		return x[i+3]
	return constraint

def odom(msg):
	global P1,P2,P3,Pc,Pr,Pb,A,b
	
	Pc = np.array(msg.data[18:21])
	Pr = np.array(msg.data[21:24])
	Pb = np.array(msg.data[24:27])
	P1 = np.array(msg.data[0:3])
	P2 = np.array(msg.data[6:9])
	P3 = np.array(msg.data[12:15])	

	A = np.array([ \
				  (-2*(Pr-P1)[:2]).tolist()+[0], \
				  (-2*(Pr-P2)[:2]).tolist()+[0], \
				  (-2*(Pr-P3)[:2]).tolist()+[0], \
				  (2*(Pr-P1)[:2]).tolist()+[0], \
				  (2*(Pr-P2)[:2]).tolist()+[0], \
				  (2*(Pr-P3)[:2]).tolist()+[0], \
				  (-2*(Pr-Pc)[:2]).tolist()+[0], \
				  (-2*(Pr-Pb)[:2]).tolist()+[0], \
				  (2*(Pr-Pc)[:2]).tolist()+[0], \
				  (2*(Pr-Pb)[:2]).tolist()+[0], \
				  [0]*2+[-1], \
				  [0]*2+[1] \
				  ])
	
	b = np.array([ \
				  np.linalg.norm((Pr-P1)[:2])**2 - d_safe_car**2, \
				  np.linalg.norm((Pr-P2)[:2])**2 - d_safe_car**2, \
				  np.linalg.norm((Pr-P3)[:2])**2 - d_safe_car**2, \
				  d_measuring**2 - np.linalg.norm((Pr-P1)[:2])**2, \
				  d_measuring**2 - np.linalg.norm((Pr-P2)[:2])**2, \
				  d_measuring**2 - np.linalg.norm((Pr-P3)[:2])**2, \
				  np.linalg.norm((Pr-Pc)[:2])**2 - d_safe_uav**2, \
				  np.linalg.norm((Pr-Pb)[:2])**2 - d_safe_uav**2, \
				  d_communication**2 - np.linalg.norm((Pr-Pc)[:2])**2, \
				  d_communication**2 - np.linalg.norm((Pr-Pb)[:2])**2, \
				  Pr[2] - height_l, \
				  height_u - Pr[2] \
				  ])
def	qpsolver():
	global range_cmd_vel
	
	cons = []
	
	for i in range (b.size):
		cons.append({'type': 'eq', 'fun': cons_maker(i)})
	for i in range (b.size):
		cons.append({'type': 'ineq', 'fun': slack(i)})
	
	ini = tuple(np.zeros(b.size + 3))
	bnds = ((-np.inf, np.inf),)*3 + ((0, np.inf),)*b.size

	optimal = minimize(object_fun, ini, method='SLSQP', bounds=bnds, constraints=cons,options={'maxiter':1000,"disp":False}).x
	#print(object_fun(optimal[:3]))

	optimal = np.clip(optimal,-0.5,0.5)

	range_cmd_vel.linear.x = optimal[0]
	range_cmd_vel.linear.y = optimal[1]
	range_cmd_vel.linear.z = optimal[2]
	
	px4_ranging.vel_control(range_cmd_vel)

if __name__ == '__main__':
	try:
		rospy.init_node('range_controller')
		px4_ranging = Px4Controller("iris_ranging")
		rospy.Subscriber('/state', Float64MultiArray, odom, queue_size=10)
		rate = rospy.Rate(100)
		while b is None:
			rate.sleep()

		while not rospy.is_shutdown():
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
