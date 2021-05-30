from threading import local
import numpy as np
import math
import numba
from matplotlib import pyplot as plt
from numba.extending import overload

# @overload(np.array)
# def np_array_ol(x):
#     if isinstance(x, numba.types.Array):
#         def impl(x):
#             return np.copy(x)
#         return impl

v_min = -1.0
v_max = 1.0
w_min = -3.0
w_max = 3.0
va = 0.5
wa = 1.0
vreso = 0.01
wreso = 0.05
robot_width = 0.4  
robot_length = 0.47  
dt = 0.1
predict_time = 4.0
alpha = 0.8
beta = 1.9
gamma = 1.0
robot_stuck_flag_cons = 0.001

# @numba.jit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64))
@numba.jit(nopython=True)
def Motion(X, u, dt):
    X[0] += u[0]*dt*math.cos(X[2])           #x方向上位置
    X[1] += u[0]*dt*math.sin(X[2])           #y方向上位置
    X[2] += u[1]*dt                     #角度变换
    X[3] = u[0]                         #速度
    X[4] = u[1]                         #角速度
    return X

@numba.jit(nopython=True)
def V_Range(X):
    Vmin_Actual = X[3]-va*dt          #实际在dt时间间隔内的最小速度
    Vmax_actual = X[3]+va*dt          #实际载dt时间间隔内的最大速度
    Wmin_Actual = X[4]-wa*dt          #实际在dt时间间隔内的最小角速度
    Wmax_Actual = X[4]+wa*dt          #实际在dt时间间隔内的最大角速度
    VW = [max(v_min,Vmin_Actual),min(v_max,Vmax_actual),max(w_min,Wmin_Actual),min(w_max,Wmax_Actual)]  #因为前面本身定义了机器人最小最大速度所以这里取交集
    return VW

# @numba.jit(numba.float64(numba.float64[:,:],numba.float64[:]))
@numba.jit(nopython=True)
def goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost + math.sqrt(dx**2 + dy**2)

# @numba.jit(numba.float64(numba.float64[:,:]))
@numba.jit(nopython=True)
def velocity_cost(trajectory):
    return v_max - trajectory[-1,3]

@numba.jit(nopython=True)
def obstacle_cost(trajectory, ob):
    MinDistance = 99999          #初始化时候机器人周围无障碍物所以最小距离设为无穷
    for i in range(len(trajectory)):           #对每一个位置点循环
        for j in range(len(Obstacle)):  #对每一个障碍物循环
            Current_Distance = math.sqrt((trajectory[i,0]-Obstacle[j,0])**2+(trajectory[i,1]-Obstacle[j,1])**2)  #求出每个点和每个障碍物距离
            theta = math.atan2(trajectory[i,1]-Obstacle[j,1], trajectory[i,0]-Obstacle[j,0]) - trajectory[i,2]
            x = (trajectory[i,0]-Obstacle[j,0])*math.cos(trajectory[i,2]) + (trajectory[i,1]-Obstacle[j,1])*math.sin(trajectory[i,2])
            y = math.sqrt(Current_Distance**2 - x**2)
            if abs(x) <= robot_length and y <= robot_width:            #如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
                return 99999
            if Current_Distance < MinDistance:
                MinDistance=Current_Distance         #得到点和障碍物距离的最小
 
    return 1.0/MinDistance

# @numba.jit(numba.float64[:,:](numba.float64[:],numba.float64[:]))
@numba.jit()
def calculate_traj(x, u):
    Traj = np.array(x)
    x_new = np.array(x)
    time = 0
    while time <= predict_time:        #一条模拟轨迹时间
        x_new = Motion(x_new, u, dt)
        Traj = np.vstack((Traj, x_new))         #一条完整模拟轨迹中所有信息集合成一个矩阵
        time += dt
    return Traj

class DWAPlanner(object):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    @numba.jit()
    def plan(x, goal, obstacles):
        vw = V_Range(x)
        best_u = [0.0, 0.0]
        min_cost = 99999
        best_trajectory = np.array(x)
        for v in np.arange(vw[0], vw[1], vreso):         #对每一个线速度循环
            for w in np.arange(vw[2], vw[3], wreso):     #对每一个角速度循环
                traj = calculate_traj(x,[v,w])

                goal_score = alpha * goal_cost(traj, goal)
                vel_score = beta * velocity_cost(traj)
                obs_score = gamma * obstacle_cost(traj, obstacles)

                score = goal_score + vel_score + obs_score

                if min_cost > score:
                    min_cost = score
                    best_u = [v, w]
                    best_trajectory = traj
                    if abs(best_u[0]) < robot_stuck_flag_cons \
                        and abs(x[3]) < robot_stuck_flag_cons and math.hypot(x[0]-goal[0],x[1]-goal[1])>1.2:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                        best_u[1] = -w_max

        return best_u, best_trajectory 

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(x, y, yaw):  # pragma: no cover
    outline = np.array([[-robot_length / 2, robot_length / 2,
                            (robot_length / 2), -robot_length / 2,
                            -robot_length / 2],
                        [robot_width / 2, robot_width / 2,
                            - robot_width / 2, -robot_width / 2,
                            robot_width / 2]])
    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), "-k")

if __name__ == "__main__":
    Obstacle = np.array([[0,10],[2,10],[4,10],[6,10],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[10,7],[10,9],[10,11],[10,13]])
    x = np.array([2,2,45*math.pi/180,0,0])                          #设定初始位置，角速度，线速度
    u = np.array([0,0])                                        #设定初始速度
    goal = np.array([8,8])                                     #设定目标位置
    global_tarj = np.array(x)
    show_animation = True
    while True:                                    
        u,current = DWAPlanner().plan(x,goal,Obstacle)
        x = Motion(x,u,dt)
        global_tarj=np.vstack((global_tarj,x))                 #存储最优轨迹为一个矩阵形式每一行存放每一条最有轨迹的信息
        if math.hypot(x[0]-goal[0],x[1]-goal[1])<=0.2:  #判断是否到达目标点
            print('Arrived')
            break

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current[:, 0], current[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.text(x[0], x[1], "v:"+str(u[0])+" w:"+str(u[1]))
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(Obstacle[:, 0], Obstacle[:, 1], "ok")
            plot_robot(x[0], x[1], x[2])
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
    
    if show_animation:
        plt.plot(global_tarj[:,0],global_tarj[:,1],'*r',Obstacle[0:3,0],Obstacle[0:3,1],'-g',Obstacle[4:9,0],Obstacle[4:9,1],'-g',Obstacle[10:13,0],Obstacle[10:13,1],'-g')  #画出最优轨迹的路线
    plt.show()