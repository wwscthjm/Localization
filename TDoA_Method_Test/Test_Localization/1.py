from array import *

#                       x        y       z        dt
anchor_A = array('f', [100     ,-50     ,0.3     ,0.5])
anchor_B = array('f', [100     ,500     ,-2      ,0.5])
anchor_C = array('f', [400     ,0       ,0.5     ,0.3])
anchor_D = array('f', [-800    ,50      ,-0.2    ,0.5])
anchor_E = array('f', [500     ,200     ,0.6     ,0.4])
anchor_F = array('f', [0       ,500     ,-0.9    ,0.5])
anchor_G = array('f', [0       ,600     ,0       ,0.9])
anchor_H = array('f', [-400    ,100     ,100     ,0.6])
anchor_I = array('f', [300     ,500     ,0.7     ,0.5])
anchor_J = array('f', [-200    ,900     ,1       ,0.5])
sensor_S = array('f', [200     ,-100    ,-1000   ,0.5])
def getSubLists(lis=[],m=2):
    allAns = []                    #用来存储所有递归中的子列表
    ans = [None for i in range(m)] #预先填充m个None,用来存储每次递归中的子列表    
    subLists(lis,m,ans,allAns)
    return allAns
def subLists(lis=[],m=0,ans=[],allAns=[]):
    # recursive function  codes
    if m==0:
        # m==0是某次递归返回的条件之一：子列表的第三个数已经选出。
        # 意味着已到达某个方向的最大递归深度
        # print('allAns is ',allAns,'before append ans:',ans)
        allAns.append(ans.copy()) 
        #这里有意思了，如果不用copy,那么ans即使已经存储在allAns，也会被其它递归方向中的ans刷新
        # print('allAns is ', allAns, 'after append ans:', ans)
        return
    length=len(lis)
    for iter in range(length-m+1):  #可以作为子列表一员的数在lis中的index
        ans[-m]=lis[iter]           #lis[iter]作为子列表倒数第m个数
        if iter+1<length:           #可以调用递归函数的条件：保证lis[iter+1:]里面还有东东才行
            subLists(lis[iter+1:],m-1,ans,allAns)
        else:
            # print('allAns is ', allAns, 'before append ans:', ans)
            allAns.append(ans.copy())
            # print('allAns is ', allAns, 'after append ans:', ans)
            return

if __name__=='__main__':
    liss=[anchor_B,anchor_C,anchor_D,anchor_E,anchor_F,anchor_G,anchor_H,anchor_I,anchor_J]
    m=2
    ans = getSubLists(liss)
    for i in range(len(ans)):
        print(str(i),':',ans[i][0])