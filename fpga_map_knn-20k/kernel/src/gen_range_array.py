
# generate the range arrays for LOAM_ODOM_KNN.
# RANGE_SPLIT 需要手动设置，后面的自动生成。

import math


CENTER_EXPAND_MAX = 128

print("static type_range_quare_hw CENTER_EXPAND_INDEX[k_CENTER_EXPAND_INDEX_MAX]", end=' = {')
for i in range(0,CENTER_EXPAND_MAX):
    if(i == 0):
        print(i, end=', ')
    elif(i == (CENTER_EXPAND_MAX-1)):
        print(-i, end=', ')
        print(i, end='')
    else:
        print(-i, end=', ')
        print(i, end=', ')
print("", end='};')
print("")        
input()

MAX_RANGE_NUM = 20
MAX_RANGE_VALUE = 60
MIN_RANGE_VALUE = 2
Rin = 1.0
Horizon_SCAN = 1800
N_SCAN = 64
LiDAR_vertical_angle = 0.47124  # 弧度，角度为 27 度
COLUMN_REGION_MAX = 64
SCAN_REGION_MAX = 64


# RANGE_SPLIT = []
RANGE_SPLIT_SQUE = []
RANGE_LOW_THRE = [0.5]
RANGE_LOW_THRE_SQUE = [0.5*0.5]
RANGE_HIGH_THRE = []
RANGE_HIGH_THRE_SQUE = []
RANGE_SEARCH_REGION_SCAN = []
RANGE_SEARCH_REGION_COLUMN = []

RANGE_SPLIT_LEVEL = [0.5, 1, 2, 4, 8, 16]
RANGE_SPLIT_LEVEL_NUM = len(RANGE_SPLIT_LEVEL)
SPLIT_GAP = MAX_RANGE_VALUE / RANGE_SPLIT_LEVEL_NUM

# print(RANGE_SPLIT_LEVEL_NUM, SPLIT_GAP)

# 32
# RANGE_SPLIT = [2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 
#                 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
#                 35, 40, 45, 50, 55, 60]
# 64
# RANGE_SPLIT = [1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
#                 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8,
#                 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8,
#                 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8,
#                 11, 12, 13, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
#                 35, 40, 45, 50, 55, 60]
# 72
RANGE_SPLIT = [1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
                5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8,
                7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8,
                9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8,
                11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 
                35, 40, 45, 50, 55, 60]
RANGE_SPLIT_SIZE = len(RANGE_SPLIT)
print(RANGE_SPLIT_SIZE)

input()

for i in range(RANGE_SPLIT_SIZE):
    RANGE_SPLIT_SQUE.append(round( RANGE_SPLIT[i] * RANGE_SPLIT[i], 2))
    RANGE_LOW_THRE.append(round( RANGE_SPLIT[i] - 1, 2) )
    RANGE_LOW_THRE_SQUE.append(round((RANGE_SPLIT[i] - 1) * (RANGE_SPLIT[i] - 1), 2))
    RANGE_HIGH_THRE.append(round(RANGE_SPLIT[i] + 1, 2))
    RANGE_HIGH_THRE_SQUE.append( round( (RANGE_SPLIT[i] + 1) * (RANGE_SPLIT[i] + 1), 2))
    
    # mid_range = (RANGE_LOW_THRE[i] + RANGE_HIGH_THRE[i])/2
    mid_range = (RANGE_LOW_THRE[i] + RANGE_LOW_THRE[i])/2
    if(mid_range < 2):
        column_region = Horizon_SCAN * Rin / (2 * 3.1415926 * mid_range)
        scan_region = N_SCAN
    else:
        column_region = Horizon_SCAN * Rin / (2 * 3.1415926 * mid_range)
        scan_region = N_SCAN / LiDAR_vertical_angle * math.atan(Rin / math.sqrt(mid_range * mid_range - Rin * Rin))

    if(column_region > COLUMN_REGION_MAX):
        column_region = COLUMN_REGION_MAX
    if(scan_region > SCAN_REGION_MAX):
        scan_region = SCAN_REGION_MAX
    
    RANGE_SEARCH_REGION_SCAN.append(round(scan_region,2))
    RANGE_SEARCH_REGION_COLUMN.append(round(column_region,2))

    # print(mid_range, scan_region, column_region, RANGE_SPLIT[i], RANGE_SPLIT_SQUE[i], RANGE_LOW_THRE[i], RANGE_HIGH_THRE[i])


# 打印 RANGE_SPLIT
print("static float RANGE_SPLIT[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_SPLIT)):
    if(i == (len(RANGE_SPLIT)-1) ):
        print(RANGE_SPLIT[i], end='')
    else:
        print(RANGE_SPLIT[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_SPLIT_SQUE
print("static float RANGE_SPLIT_SQUE[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_SPLIT_SQUE)):
    if(i == (len(RANGE_SPLIT_SQUE)-1) ):
        print(RANGE_SPLIT_SQUE[i], end='')
    else:
        print(RANGE_SPLIT_SQUE[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_LOW_THRE 注意。。这个比别的多一个数据。
print("static float RANGE_LOW_THRE[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_LOW_THRE)-1):
    if(i == (len(RANGE_LOW_THRE)-2) ):
        print(RANGE_LOW_THRE[i], end='')
    else:
        print(RANGE_LOW_THRE[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_LOW_THRE_SQUE 注意。。这个比别的多一个数据。
print("static float RANGE_LOW_THRE_SQUE[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_LOW_THRE_SQUE)-1):
    if(i == (len(RANGE_LOW_THRE_SQUE)-2) ):
        print(RANGE_LOW_THRE_SQUE[i], end='')
    else:
        print(RANGE_LOW_THRE_SQUE[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_HIGH_THRE
print("static float RANGE_HIGH_THRE[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_HIGH_THRE)):
    if(i == (len(RANGE_HIGH_THRE)-1) ):
        print(RANGE_HIGH_THRE[i], end='')
    else:
        print(RANGE_HIGH_THRE[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_HIGH_THRE_SQUE
print("static float RANGE_HIGH_THRE_SQUE[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_HIGH_THRE_SQUE)):
    if(i == (len(RANGE_HIGH_THRE_SQUE)-1) ):
        print(RANGE_HIGH_THRE_SQUE[i], end='')
    else:
        print(RANGE_HIGH_THRE_SQUE[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_SEARCH_REGION_SCAN
print("static float RANGE_SEARCH_REGION_SCAN[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_SEARCH_REGION_SCAN)):
    if(i == (len(RANGE_SEARCH_REGION_SCAN)-1) ):
        print(RANGE_SEARCH_REGION_SCAN[i], end='')
    else:
        print(RANGE_SEARCH_REGION_SCAN[i], end=', ')
print("", end='};')
print("")

# 打印 RANGE_SEARCH_REGION_COLUMN
print("static float RANGE_SEARCH_REGION_COLUMN[K_RANGE_NUM]", end=' = {')
for i in range(len(RANGE_SEARCH_REGION_COLUMN)):
    if(i == (len(RANGE_SEARCH_REGION_COLUMN)-1) ):
        print(RANGE_SEARCH_REGION_COLUMN[i], end='')
    else:
        print(RANGE_SEARCH_REGION_COLUMN[i], end=', ')
print("", end='};')
print("")


print("K_RANGE_NUM = ", RANGE_SPLIT_SIZE)