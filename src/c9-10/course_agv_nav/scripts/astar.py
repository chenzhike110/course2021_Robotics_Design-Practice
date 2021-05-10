import math
import time

import numpy as np
from numpy import random

def straight_distance(point1, point2):
    return np.sqrt(np.square(point1-point2).sum())

# 检测在地图点的碰撞半径之内是否有障碍点的存在
def collision_detect_map(map_point, position):
    collision_radius = 160
    x_min = map_point[0,0] - collision_radius
    x_max = map_point[0,0] + collision_radius
    y_min = map_point[0,1] - collision_radius
    y_max = map_point[0,1] + collision_radius
    colliset = (position[:,0]>=x_min) & (position[:,0]<=x_max) & (position[:,1]>=y_min) & (position[:,1]<=y_max)
    colliset = np.concatenate((colliset,colliset), axis=1)
    position = position[colliset]

    if position.shape[1] == 0:
        return True
    else:
        return False

#碰撞检测，无碰撞返回真
#使用圆与直线的几何关系进行判断，由于实际检测只需要检测一小段直线的碰撞因此需要进行截取
def collision_detect(p1,p2,points):
    colli_radius = 160
    
    #计算直线参数 k,b
    if p2[0,0]-p1[0,0]==0:      #！！！！！！！
        return False
    k = (p2[0,1]-p1[0,1])/(p2[0,0]-p1[0,0])
    b = (p2[0,0]*p1[0,1]-p1[0,0]*p2[0,1])/(p2[0,0]-p1[0,0])
    
    #选取线段范围内的圆
    x_min = min(p1[0,0],p2[0,0]) - colli_radius
    x_max = max(p1[0,0],p2[0,0]) + colli_radius
    y_min = min(p1[0,1],p2[0,1]) - colli_radius 
    y_max = max(p1[0,1],p2[0,1]) + colli_radius
    mask = (points[:,0]>=x_min) & (points[:,0]<=x_max) & (points[:,1]>=y_min) & (points[:,1]<=y_max)
    mask = np.concatenate((mask,mask), axis=1)
    points = points[mask]
    points_row = points.shape[1]/2
    points = points.reshape((int(points_row), 2))   # 此处points开始为行向量，需要reshape成n*2矩阵

    #碰撞检测
    if(points.shape[0] == 0):
        #如果直线碰撞半径范围内，一个圆都没有
        return True
    else:
        #计算范围内每个圆心到直线的距离
        dist = np.abs(points[:,1]-k*points[:,0]-b)/math.sqrt(1+k*k)
        if(dist.min()>colli_radius):
            #大于碰撞半径无碰撞
        # no collision
            return True
        else: 
            return False

class PathPlan(object):
    def __init__(self, position, debugger, num_point, alpha, beta):
        self.map_height = 3000
        self.map_width = 4500
        self.position = np.mat(position)

        myposition = self.vision.get_myposition()
        print(myposition)
        self.start_point = np.mat([myposition[0], myposition[1]])    # 左下到右上
        self.end_point = np.mat([-2400, -1500])
        self.map = np.append(self.start_point, self.end_point, axis=0)
        self.map = np.append(self.map, [[0],[10000]], axis=1)
        self.astar_tree = np.append(self.start_point, [[0]], axis=1)
        self.num_points = num_point    # 构建具有N个点的连通图（可调参数）
        self.alpha = alpha    # 是否接受地图点参数（可调参数）
        self.beta = beta
        self.openlist = np.mat(np.zeros(5))
        self.closelist = np.mat(np.zeros(5))
        self.debugger = debugger
        debugger.draw_circle(self.start_point[0,0],self.start_point[0,1])
        debugger.draw_circle(self.end_point[0,0],self.end_point[0,1])
    
    def random(self):
        self.start_point = np.mat(random.randint(-self.map_height,0,size=(1,2)))
        self.end_point = np.mat(random.randint(0,self.map_height,size=(1,2)))
    
    # 以是否生成足够多的点（分辨率是否足够高）
    # 和生成点是否最简（是否基本在起始点到终止点的直线周围）
    # 作为概率依据接受随机点的生成
    def accept_point(self, map_point, n):
        dist1 = straight_distance(map_point, self.start_point)
        dist2 = straight_distance(map_point, self.end_point)
        dist = straight_distance(self.start_point, self.end_point)
        P = self.alpha*( 1 - n/self.num_points) + self.beta*math.pow( 100, math.pow(dist, 2)/math.pow(dist1+dist2, 2) - 1 )
        if random.random() < P:
            return True
        else:
            return False

    def build_map(self):
        n = 0   # 已生成的点的个数
        while n <= self.num_points:
            map_point = np.mat(random.randint(-self.map_height,self.map_height,size=(1,2)))
            if collision_detect_map(map_point, self.position):
                if self.accept_point(map_point, n):
                    n = n + 1
                    # print(n)
                    map_point = np.append(map_point, [[10000]], axis=1)
                    self.map = np.append(self.map, map_point, axis=0)
                    self.debugger.draw_point(map_point[0,0], map_point[0,1])

    def IsStartPoint(self, p):
        if (p[0,0] == self.start_point[0,0]) and (p[0,1] == self.start_point[0,1]):
            return True
        else:
            return False

    def IsEndPoint(self, p):
        if (p[0,0] == self.end_point[0,0]) and (p[0,1] == self.end_point[0,1]):
            return True
        else:
            return False

    def IsInPointList(self, p, point_list):
        meetset = (point_list[:,0]==p[0,0]) & (point_list[:,1]==p[0,1])
        meetset = np.concatenate((meetset,meetset), axis=1)
        point_list = point_list[meetset]
        if point_list.shape[1] == 0:
            return False
        else:
            return True

    def IsInOpenList(self, p):
        return self.IsInPointList(p[0,0:2], self.openlist[:,0:2])

    def IsInCloseList(self, p):
        return self.IsInPointList(p[0,0:2], self.closelist[:,0:2])

    def IsInAstarTree(self, p):
        return self.IsInPointList(p[0,0:2], self.astar_tree[:,0:2])
    
    def FindTreeIndex(self, p):
        meetset = (self.astar_tree[:,0]==p[0,0]) & (self.astar_tree[:,1]==p[:,1])
        index = np.where(meetset==True)
        return int(index[0])
    
    def NeighborPoint(self, p, map):
        # 提取指定范围内的临近点
        neighradius = 1000
        x_min = p[0,0]-neighradius
        x_max = p[0,0]+neighradius
        y_min = p[0,1]-neighradius
        y_max = p[0,1]+neighradius
        neighset = (map[:,0]>=x_min) & (map[:,0]<=x_max) & (map[:,1]>=y_min) & (map[:,1]<=y_max)
        neighset = np.concatenate( (np.concatenate((neighset, neighset), axis=1), neighset), axis=1 )
        map = map[neighset]
        map_row = map.shape[1]/3
        map = map.reshape((int(map_row), 3))

        # 找出5个无碰撞点
        i = 0
        while True:
            if collision_detect(p[0,0:2], map[i,0:2], self.position):
                neighborpoint = map[i,:]
                n = 1
                break
            i = i+1
        i = i+1
        while n < 5 and i < np.size(map, 0):
            if collision_detect(p[0,0:2], map[i,0:2], self.position):
                neighborpoint = np.append(neighborpoint, map[i,:], axis=0)
                n = n+1
            i = i+1
        return neighborpoint

    def NeighborPoint_update(self, p, map):
        # 依照与p点距离大小进行排序
        row = map.shape[0]
        column = map.shape[1]
        dist = np.mat(np.arange(row*column).reshape(row, column))
        dist[:,0] = map[:,0]-np.mat(p[0,0]*np.ones(row)).T
        dist[:,1] = map[:,1]-np.mat(p[0,1]*np.ones(row)).T
        dist = np.square(dist)
        distance = dist[:,0]+dist[:,1]
        sort_index = np.argsort(distance, axis=0)

        # 找出10个无碰撞点
        i = 0
        index = int(np.where(sort_index==i)[0])
        while True:
            if collision_detect(p[0,0:2], map[index,0:2], self.position):
                neighborpoint = map[index,:]
                n = 1
                break
            i = i+1
            index = int(np.where(sort_index==i)[0])
        i = i+1
        index = int(np.where(sort_index==i)[0])
        while n < 5:
            if collision_detect(p[0,0:2], map[index,0:2], self.position):
                neighborpoint = np.append(neighborpoint, map[index,:], axis=0)
                n = n+1
            i = i+1
            index = int(np.where(sort_index==i)[0])
        return neighborpoint

    def UpdateMapCost(self, p):
        meetset = (self.map[:,0]==p[0,0]) & (self.map[:,1]==p[:,1])
        index = int(np.where(meetset==True)[0])
        self.map[index,2] = p[0,2]
    
    def UpdateOpenList(self, p, h, f):
        meetset = (self.openlist[:,0]==p[0,0]) & (self.openlist[:,1]==p[:,1])
        index = int(np.where(meetset==True)[0])
        self.openlist[index,2:5] = [p[0,2], h, f]

    def HeuristicCost(self, p):
        return straight_distance(p[0, 0:2], self.end_point)

    def BuildPath(self, p):
        p_index = self.FindTreeIndex(p)
        p = self.astar_tree[p_index,:]
        path = []
        while True:
            path.insert(0, p)
            if self.IsStartPoint(p):
                break
            else:
                parent = self.astar_tree[p[0,2],:]
                print(parent.shape)
                print(parent)
                self.debugger.draw_line(p[0,0], p[0,1], parent[0,0], parent[0,1], 'white')
                p = parent
        print('finish')
        return True
    
    def ProcessPoint(self, p, parent):
        tg = parent[0,2] + straight_distance(p[0,0:2], parent[0,0:2])
        self.debugger.draw_line(p[0,0]-60, p[0,1], p[0,0]+60, p[0,1], 'blue')
        if not ( self.IsInCloseList(p) and ( tg >= p[0,2] ) ):
            # print('go 1')
            if ( not self.IsInOpenList(p) ) or ( tg < p[0,2] ):
                # print('go 2')
                parent_index = self.FindTreeIndex(parent)
                if self.IsInAstarTree(p):
                    p_index = self.FindTreeIndex(p)
                    self.astar_tree[p_index, 2] = parent_index
                else:
                    self.astar_tree = np.append(self.astar_tree, [[p[0,0], p[0,1], parent_index]], axis=0)
                # self.debugger.draw_line(p[0,0]-60, p[0,1], p[0,0]+60, p[0,1], 'blue')
                # self.debugger.draw_line(p[0,0], p[0,1], parent[0,0], parent[0,1], 'blue')
                # 更新临节点g(n)=tg，并计算h(n)和f(n)
                p[0,2] = tg
                self.UpdateMapCost(p)
                h = self.HeuristicCost(p)
                f = p[0,2] + h
                if self.IsInOpenList(p):
                    self.UpdateOpenList(p, h, f)
                else:
                    node = np.mat([p[0,0], p[0,1], p[0,2], h, f])
                    # print(self.openlist.shape)
                    # print(node.shape)
                    self.openlist = np.append(self.openlist, node, axis=0)
        #             print('openlist+1')
        # print('process over')
     
    def Astar(self):
        # 创建连通图
        self.build_map()
        
        # print(self.map.shape)
        # 将起始点放入openlist中，并计算g(n),h(n),f(n)
        self.openlist[0,0] = self.start_point[0,0]
        self.openlist[0,1] = self.start_point[0,1]
        self.openlist[0,2] = 0
        self.openlist[0,3] = straight_distance(self.start_point, self.end_point)
        self.openlist[0,4] = self.openlist[0,2] + self.openlist[0,3]

        while True:
            # 取出openlist中f(n)最小的节点，记为p
            if self.openlist.shape[0] == 0:
                return False
            minindex = np.argmin(self.openlist, axis = 0)
            p = np.mat(self.openlist[minindex[0,4], 0:3])
            
            # 判断p是否为终止节点，如是则回溯
            if self.IsEndPoint(p):
                return self.BuildPath(p)
            
            # 将节点p移出openlist，放入closelist中
            self.closelist = np.append(self.closelist, self.openlist[minindex[0,4], :], axis=0)
            self.openlist = np.delete(self.openlist, minindex[0,4], axis=0)
            # 得到临近点
            neighborpoint = self.NeighborPoint(p, self.map)
            # neighborpoint = self.NeighborPoint_update(p, self.map)
            # 处理所有临近点
            for e in neighborpoint:
                self.ProcessPoint(e, p)
            
            
            # print('astar over')
                
