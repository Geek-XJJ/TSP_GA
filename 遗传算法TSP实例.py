import random
from matplotlib import pyplot as plt
#   定义一个20*3的矩阵，分别代表20个城市以及各自的x和y坐标
city = [[1, 54, 51],
        [2, 22, 67],
        [3, 4, 26],
        [4, 57, 62],
        [5, 12, 42],
        [6, 38, 32],
        [7, 61, 31],
        [8, 21, 14],
        [9, 79, 10],
        [10, 29, 15],
        [11, 52, 33],
        [12, 82, 27],
        [13, 21, 96],
        [14, 37, 91],
        [15, 9, 75],
        [16, 74, 76],
        [17, 17, 27],
        [18, 65, 94],
        [19, 70, 71],
        [20, 41, 98]]
number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]    # 城市序号


def first_solution(cnt: int) -> list[list[int]]:  # 产生初始种群，cnt*20的矩阵
    result: list[list[int]] = []
    for i in range(cnt):
        result.append(random.sample(number, 20))
    return result


def dist_comput(route: list[int]) -> float:    # 计算某个路线的总距离
    total_dist = 0  # 某个解的总距离
    full_path = route + [route[0]]
    for j in range(len(full_path)-1):
        x1 = city[full_path[j]-1][1]
        y1 = city[full_path[j]-1][2]
        x2 = city[full_path[j+1]-1][1]
        y2 = city[full_path[j+1]-1][2]
        dist = (abs(x1-x2) ** 2 + abs(y1-y2) ** 2) ** 0.5
        if dist == 0:
            dist = 0.1
        total_dist += dist
    if total_dist == 0:
        total_dist = 1
    return total_dist


def fitness(solu: list[list[int]]):   # 计算某个解集所有解的适应度，返回归一化选择概率
    fitness_list: list[float] = []   # 存储每个解的适应度，为每个解总距离的倒数
    for i in range(len(solu)):
        total_dist = dist_comput(solu[i])
        fitness_list.append(1/total_dist)
    prob_one = [item/sum(fitness_list) for item in fitness_list]    # 归一化选择概率
    return prob_one


def remove_common_elements(sec0, sec1):    # 去除两个列表sec0和sec1中相同的元素,并返回两个新的列表
    # 转换为集合以快速查找公共元素
    set0 = set(sec0)
    set1 = set(sec1)
    # 计算公共元素
    common_elements = set0 & set1
    # 去除公共元素后构建新的列表
    new_sec0 = [x for x in sec0 if x not in common_elements]
    new_sec1 = [x for x in sec1 if x not in common_elements]
    return new_sec0, new_sec1


def select_crossover(solution: list[list[int]]) -> list[list[int]]:    # 选择并交叉父母信息，返回新的种群
    new_group: list[list[int]] = []
    one_prob = fitness(solution)
    for i in range(1, len(one_prob)):   # 叠加选择概率，创建轮盘赌
        one_prob[i] += one_prob[i-1]
    for n in range(len(solution)//2):
        parents: list[list[int]] = []    # 存储父母
        for i in range(2):  # 生成随机数来进行轮盘赌，确定父母
            rand = random.uniform(0, 1)
            for k in range(len(one_prob)):
                if k == 0:
                    if rand < one_prob[k]:
                        parents.append(solution[k])
                else:
                    if one_prob[k-1] <= rand < one_prob[k]:
                        parents.append(solution[k])
        p0 = parents[0][:]
        p1 = parents[1][:]
        rand = random.randint(0, 14)
        sec0 = p0[rand:rand+5][:]
        sec1 = p1[rand:rand+5][:]
        new_sec0, new_sec1 = remove_common_elements(sec0, sec1)
        for i in new_sec0:
            for j in range(len(p1)):
                if p1[j] == i:
                    p1[j] = new_sec1[new_sec0.index(i)]    # 去重以保证城市唯一性
        son1 = p1[:rand] + sec0 + p1[rand+5:]   # 子体son1

        for i in new_sec1:
            for j in range(len(p0)):
                if p0[j] == i:
                    p0[j] = new_sec0[new_sec1.index(i)]    # 去重以保证城市唯一性
        son0 = p0[:rand] + sec1 + p0[rand+5:]   # 子体son0
        # 将新产生的2个子体加入新种群
        new_group.append(son1)
        new_group.append(son0)
    return new_group


def mutation(group: list[list[int]], rate: float) -> list[list[int]]:  # 生成随机数进行随机某个点位的变异，返回变异后的新种群
    new_group: list[list[int]] = []
    for i in range(len(group)):
        rand = random.uniform(0, 1)
        copy_1 = group[i][:]
        if rand < rate:    # 生成的随机数小于变异概率，则进行变异操作
            k = random.randint(0, len(copy_1)-2)
            copy_1[k], copy_1[k+1] = copy_1[k+1], copy_1[k]    # 变异规则为:使路线中下标为k与k+1的城市交换
            new_group.append(copy_1)
        else:
            new_group.append(group[i][:])
    return new_group


def print_route(road: list[int]):
    for i in range(len(road)):
        print(f"{road[i]} -> ", end="")
    print(f"{road[0]}")


popu_scale = 100    # 种群规模
gen_num = 1000   # 迭代次数
middle_solu = first_solution(popu_scale)  # 中间解,这是第一代，即初始种群
print(f"初始路线为:")
print_route(middle_solu[0])
print(f"总距离:{dist_comput(middle_solu[0])}")
good_son = []   # 存储每一代种群中最优秀的个体及其归一化选择率,总共有gen_num个数据
for i in range(gen_num):
    middle_solu = select_crossover(middle_solu)
    middle_solu = mutation(middle_solu, 0.2)
    chance = fitness(middle_solu)
    good_index = chance.index(max(chance))  # 新种群中最优个体的下标
    good_son.append([middle_solu[good_index], chance[good_index]])
good_son.sort(key=lambda x: x[1], reverse=True)
print(f"进化{gen_num}代后,给出的近似最优路线为:")
print_route(good_son[0][0])
print(f"总距离:{dist_comput(good_son[0][0])}")
# 画图数据准备
final_route = good_son[0][0][:]
x_coords = []   # 最优路线的x坐标
y_coords = []   # 最优路线的y坐标
for i in range(len(final_route)):
    x_coords.append(city[final_route[i]-1][1])
for i in range(len(final_route)):
    y_coords.append(city[final_route[i]-1][2])
# 画图
plt.scatter(x_coords, y_coords)
# 绘制带箭头的线段，按顺序相连并首尾相连
for i in range(len(x_coords) - 1):
    # 绘制普通线段（除了首尾相连的）
    plt.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]), xytext=(x_coords[i], y_coords[i]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

# 绘制首尾相连的线段，并加粗箭头以突出显示
plt.annotate('', xy=(x_coords[0], y_coords[0]), xytext=(x_coords[-1], y_coords[-1]),
             arrowprops=dict(facecolor='red', arrowstyle='-|>', connectionstyle='arc3,rad=0.3'))

# 突出显示起点
plt.scatter([x_coords[0]], [y_coords[0]], color='red', s=100)  # s参数控制点的大小

# 设置图形窗口大小
plt.gcf().set_size_inches(7.5, 4.5)

# 显示图形
plt.show()
