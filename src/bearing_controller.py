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

P1,P2,P3,Pc,Pr,Pb,A,b = None,None,None,None,None,None,None,None
bearing_cmd_vel = Twist()
sigma_u,sigma_v,sigma_ranging,sigma_bearing,sigma_alpha = 0.007,0.007,0.01,0.01,0.01
height_l = 0.3
height_u = 1.8
d_safe_car = 0.7
d_measuring = 2.2
d_safe_uav = 0.7
d_communication = 20
time_last,dt = 0,0

class Objective(ElementwiseProblem):

	def __init__(self):
		super().__init__(n_var=3,n_obj=6,n_constr=b.size, \
						 xl=np.array([-0.3]*3),xu=np.array([0.3]*3))

	def _evaluate(self, x, out, *args, **kwargs):
		
		r1b = np.array([P1[0] - (Pb[0] + dt*x[0]),P1[1] - (Pb[1] + dt*x[1]),P1[2] - (Pb[2] + dt*x[2])])
		r2b = np.array([P2[0] - (Pb[0] + dt*x[0]),P2[1] - (Pb[1] + dt*x[1]),P2[2] - (Pb[2] + dt*x[2])])
		r3b = np.array([P3[0] - (Pb[0] + dt*x[0]),P3[1] - (Pb[1] + dt*x[1]),P3[2] - (Pb[2] + dt*x[2])])


		f1_b = 1/np.linalg.norm(r1b[:2])**2
		f1_a = 1/r1b[2]**2

		f2_b = 1/np.linalg.norm(r2b[:2])**2
		f2_a = 1/r2b[2]**2

		f3_b = 1/np.linalg.norm(r3b[:2])**2
		f3_a = 1/r3b[2]**2

		out["F"] = [1/f1_b, 1/f1_a, 1/f2_b, 1/f2_a, 1/f3_b, 1/f3_a]
		out["G"] = (A.dot(x.reshape((3,1))) - b).reshape(12,).tolist()

def odom(msg):
	global P1,P2,P3,Pc,Pr,Pb,A,b
	
	Pc = np.array(msg.data[18:21])
	Pr = np.array(msg.data[21:24])
	Pb = np.array(msg.data[24:27])
	P1 = np.array(msg.data[0:3])
	P2 = np.array(msg.data[6:9])
	P3 = np.array(msg.data[12:15])

	A = np.array([ \
				  (-2*(Pb-P1)[:2]).tolist()+[0], \
				  (-2*(Pb-P2)[:2]).tolist()+[0], \
				  (-2*(Pb-P3)[:2]).tolist()+[0], \
				  (2*(Pb-P1)[:2]).tolist()+[0], \
				  (2*(Pb-P2)[:2]).tolist()+[0], \
				  (2*(Pb-P3)[:2]).tolist()+[0], \
				  (-2*(Pb-Pc)[:2]).tolist()+[0], \
				  (-2*(Pb-Pr)[:2]).tolist()+[0], \
				  (2*(Pb-Pc)[:2]).tolist()+[0], \
				  (2*(Pb-Pr)[:2]).tolist()+[0], \
				  [0]*2+[-1], \
				  [0]*2+[1], \
				  ])
	
	b = np.array([ \
				  [np.linalg.norm((Pb-P1)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pb-P2)[:2])**2 - d_safe_car**2], \
				  [np.linalg.norm((Pb-P3)[:2])**2 - d_safe_car**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P1)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P2)[:2])**2], \
				  [d_measuring**2 - np.linalg.norm((Pb-P3)[:2])**2], \
				  [np.linalg.norm((Pb-Pc)[:2])**2 - d_safe_uav**2], \
				  [np.linalg.norm((Pb-Pr)[:2])**2 - d_safe_uav**2], \
				  [d_communication**2 - np.linalg.norm((Pb-Pc)[:2])**2], \
				  [d_communication**2 - np.linalg.norm((Pb-Pr)[:2])**2], \
				  [Pb[2] - height_l], \
				  [height_u - Pb[2]] \
				  ])
	
def	qpsolver():
	global bearing_cmd_vel,time_last,dt
	
	objective = Objective()
	algorithm = NSGA2(pop_size=20,n_offsprings=None,sampling=get_sampling("real_random"), \
					  crossover=get_crossover("real_sbx", prob=0.9, eta=15), \
					  mutation=get_mutation("real_pm", eta=20), eliminate_duplicates=True)
	termination = get_termination("n_gen", 8)
	dt = rospy.Time.now().to_sec() - time_last
	res = minimize(objective, algorithm, termination, seed=10, save_history=False, verbose=True, return_least_infeasible=True)
	
	tmp = np.cumsum(res.F,axis=1)
	num_opt = np.argmin(tmp[:,5])
	
	bearing_cmd_vel.linear.x = res.X[num_opt,0]
	bearing_cmd_vel.linear.y = res.X[num_opt,1]
	bearing_cmd_vel.linear.z = res.X[num_opt,2]
	
	px4_bearing.vel_control(bearing_cmd_vel)
	time_last = rospy.Time.now().to_sec()

if __name__ == '__main__':
	try:
		rospy.init_node('controller')
		px4_bearing = Px4Controller("iris_bearing")
		rospy.Subscriber('/state', Float64MultiArray, odom, queue_size=10)
		rate = rospy.Rate(100)
		while b is None:
			rate.sleep()

		while not rospy.is_shutdown():
			qpsolver()
			rate.sleep()
	except rospy.ROSInterruptException:
		pass
