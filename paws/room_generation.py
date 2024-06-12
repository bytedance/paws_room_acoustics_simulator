import numpy as np
import random
import matplotlib.pyplot as plt
import math
import cv2

def get_next_pos(x,y):

    direction = random.randint(0,3)

    if direction == 0:
        return (x-1,y)
    
    elif direction == 1:
        return (x,y-1)
    
    elif direction == 2:
        return (x+1,y)
    
    else:
        return(x,y+1)


def draw_wall(wall_set,room_pos,grid_data,padding_coef=0.1,wall_coef=0.1,class_id=1):
    #画墙壁

    room_size = grid_data.shape[0]
    pad_size = math.floor(room_size * padding_coef)
    room_nopad = room_size - pad_size*2

    x_range,y_range= room_pos.max(0) - room_pos.min(0)
    x_min, y_min = room_pos.min(0)

    size_room_x = math.floor(room_nopad / (x_range+1))
    size_room_y = math.floor(room_nopad / (y_range+1))

    size_wall_x = math.floor(size_room_x*wall_coef/2)
    size_wall_y = math.floor(size_room_x*wall_coef/2)

    # print("size_room_x = ",size_room_x)
    # print("size_room_y = ",size_room_y)
    # print("size_wall_x = ",size_wall_x)
    # print("size_wall_y = ",size_wall_y)


    for wall in wall_set:

        # print("wall = ",wall)
        x_1 = wall[0][0] - x_min
        y_1 = wall[0][1] - y_min

        x_2 = wall[1][0] - x_min
        y_2 = wall[1][1] - y_min

        # print("x_1,y_1 = ",x_1,y_1)
        # print("x_2,y_2 = ",x_2,y_2)
        x_1 *= size_room_x
        y_1 *= size_room_y
        x_2 *= size_room_x
        y_2 *= size_room_y

        x_1 = math.floor(x_1)
        y_1 = math.floor(y_1)
        x_2 = math.floor(x_2)
        y_2 = math.floor(y_2)

        x_1 -= size_wall_x
        x_2 += size_wall_x
        y_1 -= size_wall_x
        y_2 += size_wall_y

        x_1 += pad_size
        x_2 += pad_size
        y_1 += pad_size
        y_2 += pad_size

        # print("x_1,y_1 = ",x_1,y_1)
        # print("x_2,y_2 = ",x_2,y_2)
        # print("-----")

        for x_t in range(max(x_1,pad_size),min(x_2,room_nopad+pad_size)):
            for y_t in range(max(y_1,pad_size),min(y_2,room_nopad+pad_size)):

                grid_data[x_t][y_t] = class_id

    return grid_data


def random_merge_room(room_list:list,wall_set:set,keep_prob:float):

    room_group = []

    for room in room_list:
        room_group.append([room])


    while len(wall_set) > 0:

        wall = wall_set.pop()

        p1 = wall[0]
        p2 = wall[1]

        if p1[0] == p2[0]:

            room1 = [p1[0]-1,p1[1]]
            room2 = [p1[0],p1[1]]

        else:

            room1 = [p1[0],p1[1]-1]
            room2 = [p1[0],p1[1]]

        # print("p1 = ",p1)
        # print("p2 = ",p2)
        # print("room1 = ",room1)
        # print("room2 = ",room2)

        index_1 = -1
        index_2 = -1

        for i in range(len(room_group)):
            if room1 in room_group[i]:
                index_1 = i
        for j in range(len(room_group)):
            if room2 in room_group[j]:
                index_2 = j

        if index_1 != index_2:
            room_group[index_1] = room_group[index_1] + room_group[index_2]
            room_group.remove(room_group[index_2])

        # print("index1 = ", index_1)
        # print("index2 = ", index_2)
        # print(room_group)
        # print("------------------")

        if (len(room_group) == 1):
            
            if random.random() < keep_prob:
                break

    return wall_set
    

def get_valid_mask(room_pos,grid_data,padding_coef=0.1):
    
    room_size = grid_data.shape[0]
    pad_size = math.floor(room_size * padding_coef)
    room_nopad = room_size - pad_size*2


    valid_mask = [[0]*room_size]*room_size
    valid_mask = np.array(grid_data)

    x_range,y_range= room_pos.max(0) - room_pos.min(0)
    x_min, y_min = room_pos.min(0)

    size_room_x = math.floor(room_nopad / (x_range+1))
    size_room_y = math.floor(room_nopad / (y_range+1))

    # print("size_room_x = ",size_room_x)
    # print("size_room_y = ",size_room_y)

    for pos in room_pos:
        x = pos[0] - x_min
        y = pos[1] - y_min

        x *= size_room_x
        y *= size_room_y

        x = math.floor(x)
        y = math.floor(y)

        x += pad_size
        y += pad_size


        for x_t in range(x,x + size_room_x):
            for y_t in range(y, y + size_room_y):

                valid_mask[x_t][y_t] = 1

    return valid_mask


def try_add_obstacle(grid_data,valid_mask,class_id,minimun_gap = 10,max_attempt=100):

    x_range = grid_data.shape[0]
    y_range = grid_data.shape[1]


    x_size = random.randint(2,math.floor(x_range/10))
    y_size = random.randint(2,math.floor(y_range/10))

    x_pos = random.randint(x_size + minimun_gap, x_range - x_size - minimun_gap)
    y_pos = random.randint(y_size + minimun_gap, x_range - y_size - minimun_gap)

    # print(x_pos,y_pos)
    # print(x_size,y_size)

    while max_attempt > 0:

        if valid_mask[x_pos][y_pos] == 1:
            break
        
        x_pos = random.randint(x_size + minimun_gap, x_range - x_size - minimun_gap)
        y_pos = random.randint(y_size + minimun_gap, x_range - y_size - minimun_gap)

        max_attempt -= 1


    for x in range(x_pos - x_size - minimun_gap, x_pos + x_size + minimun_gap):
        for y in range(y_pos - y_size - minimun_gap, y_pos + y_size + minimun_gap):

            if grid_data[x][y] != 0:

                return grid_data
        
    for x in range(x_pos - x_size, x_pos + x_size):
        for y in range(y_pos - y_size, y_pos+y_size):

            grid_data[x][y] = class_id

    return grid_data


def shoe_box_pipeline(room_size = 256,
                      padding_coef = 0.1,
                      wall_coef = 0.1,
                      max_subroom=8,
                      max_obstacle=10,
                      class_id_range=[1,7],
                      wall_class_choice=None
                      ):

    room_pos = set()

    init_pos = (0,0)
    spawn_pos = init_pos

    for _ in range(max_subroom):
        room_pos.add(spawn_pos)
        spawn_pos = get_next_pos(spawn_pos[0],spawn_pos[1])


    # print(room_pos)

    #根据block组生成边缘墙壁（不会在接下来的流程中被消除）和内部墙壁
    outer_wall_set = set()
    inner_wall_set = set()

    for pos in room_pos:

        if (pos[0]-1,pos[1]) in room_pos:
            inner_wall_set.add(((pos[0],pos[1]),(pos[0],pos[1]+1)))
        else:
            outer_wall_set.add(((pos[0],pos[1]),(pos[0],pos[1]+1)))


        if (pos[0]+1,pos[1]) in room_pos:
            inner_wall_set.add(((pos[0]+1,pos[1]),(pos[0]+1,pos[1]+1)))
        else:
            outer_wall_set.add(((pos[0]+1,pos[1]),(pos[0]+1,pos[1]+1)))


        if (pos[0],pos[1]-1) in room_pos:
            inner_wall_set.add(((pos[0],pos[1]),(pos[0]+1,pos[1])))
        else:
            outer_wall_set.add(((pos[0],pos[1]),(pos[0]+1,pos[1])))

        if (pos[0],pos[1]+1) in room_pos:
            inner_wall_set.add(((pos[0],pos[1]+1),(pos[0]+1,pos[1]+1)))
        else:
            outer_wall_set.add(((pos[0],pos[1]+1),(pos[0]+1,pos[1]+1)))


    # print(inner_wall_set)
    # print(outer_wall_set)

    #构建坐标block组
    room_pos_new = []

    for (x,y) in room_pos: 
        room_pos_new.append([x,y]) 

    room_pos_new = np.array(room_pos_new)
    # print(room_pos_new)


    #randomly break wall
    inner_wall_set = random_merge_room(room_pos_new.tolist(),inner_wall_set,0.8)


    #初始化grid
    # room_size = 256

    grid_data = [[0]*room_size]*room_size
    grid_data = np.array(grid_data)


    #画墙壁
    # padding_coef = 0.1
    # wall_coef = 0.1
    
    if wall_class_choice != None:
        wall_class_id = random.choice(wall_class_choice)

    else:
        wall_class_id = random.randint(class_id_range[0],class_id_range[1])


    grid_data = draw_wall(inner_wall_set,room_pos_new,grid_data,padding_coef,wall_coef,wall_class_id)
    grid_data = draw_wall(outer_wall_set,room_pos_new,grid_data,padding_coef,wall_coef,wall_class_id)

    valid_mask = get_valid_mask(room_pos_new,grid_data,padding_coef)


    # # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(grid_data), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()


    # # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(valid_mask), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()


    #生成障碍物

    for _ in range(max_obstacle):

        class_id =  random.randint(class_id_range[0],class_id_range[1])
        grid_data = try_add_obstacle(grid_data,valid_mask,class_id)


    # # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(grid_data), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()


    return grid_data,valid_mask



def uniform_random(left, right, size=None):
    """
    generate uniformly distributed random numbers in [left, right)

    Parameters:
    -----------
    left: a number
        left border of random range
    right: a number
        right border of random range
    size: a number or a list/tuple of numbers
        size of output

    Returns:
    --------
    rand_nums: ndarray
        uniformly distributed random numbers
    """
    rand_nums = (right - left) * np.random.random(size) + left
    return rand_nums


def random_polygon(edge_num, center, radius_range):
    """
    generate points to construct a random polygon

    Parameters:
    -----------
    edge_num: a number
        edge numbers of polygon
    center: a list/tuple contain two numbers
        center of polygon
    radius_range: a list/tuple containing two numbers
        range of distances from center to polygon vertices

    Returns:
    --------
    points: ndarray
        points that can construct a random polygon
    """
    angles = uniform_random(0, 2 * np.pi, edge_num)
    angles = np.sort(angles)
    random_radius = uniform_random(radius_range[0], radius_range[1], edge_num)
    x = np.cos(angles) * random_radius
    y = np.sin(angles) * random_radius
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    points = np.concatenate([x, y], axis=1)
    points += np.array(center)

    return points


def get_line_func(x_1,y_1,x_2,y_2):

    A = y_2-y_1
    B = x_1-x_2
    C = x_2*y_1 - x_1*y_2

    return A,B,C


def get_dist(A,B,C,x_1,y_1):

    dist = math.fabs(A*x_1+B*y_1+C) / math.sqrt(A*A + B*B)

    return dist


def draw_polygon(grid_data,points,line_size,class_id):


    for i in range(points.shape[0]):

        point1 = points[i]
        point2 = points[i-1]

        print(point1)
        print(point2)

        x_min = math.floor(min(point1[0],point2[0]))
        x_max = math.ceil(max(point1[0],point2[0]))
        y_min = math.floor(min(point1[1],point2[1]))
        y_max = math.ceil(max(point1[1],point2[1]))


        print(x_min,x_max,y_min,y_max)

        A,B,C = get_line_func(point1[0],point1[1],point2[0],point2[1])

        for x in range(x_min-line_size,x_max+line_size):
            for y in range(y_min-line_size,y_max+line_size):

                dist = get_dist(A,B,C,x+0.5,y+0.5)

                if dist < line_size:
                    grid_data[x][y] = class_id

    
    return grid_data


def polygon_pipeline(room_size = 256,
                     padding_coef = 0.1,
                     max_obstacle=10,
                     edge_n_range=[5,10],
                     class_id_range=[1,7],
                     wall_class_choice=None
                    ):
    
    #初始化grid
    grid_data = [[0]*room_size]*room_size
    grid_data = np.array(grid_data)

    #画墙壁
    padding_coef = 0.1

    room_size = grid_data.shape[0]
    pad_size = math.floor(room_size * padding_coef)
    room_nopad = room_size - pad_size*2

    edge_num = random.randint(edge_n_range[0],edge_n_range[1])



    while True:
        points = random_polygon(edge_num, [80, 80], [60, 80])

        if np.min(points.max(0) - points.min(0)) > 40:
            break

        

    points -= (points.max(0) + points.min(0))/2

    x_range,y_range = points.max(0) - points.min(0)
    scale = room_nopad / max(x_range,y_range)
    points *= scale
    points += room_size/2



    if wall_class_choice != None:
        wall_class_id = random.choice(wall_class_choice)

    else:
        wall_class_id = random.randint(class_id_range[0],class_id_range[1])


    grid_data = draw_polygon(grid_data,points,3,wall_class_id)


    # # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(grid_display), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()


    valid_mask = [[0]*room_size]*room_size
    valid_mask = np.array(valid_mask)

    valid_mask = cv2.fillPoly(valid_mask,np.int32([points]),1)
    valid_mask = np.transpose(valid_mask)

    # # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(valid_mask), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()

    #生成障碍物

    for _ in range(max_obstacle):

        class_id =  random.randint(class_id_range[0],class_id_range[1])
        grid_data = try_add_obstacle(grid_data,valid_mask,class_id,5)


    # show demo of sampled area
    # plt.figure()
    # plt.imshow(np.squeeze(grid_data), aspect='equal', cmap='gray')
    # plt.xlabel('x-position [m]')
    # plt.ylabel('y-position [m]')
    # plt.title('generated room')
    # plt.show()

    return grid_data,valid_mask



if __name__ == "__main__":
    # shoe_box_pipeline()

    polygon_pipeline()

