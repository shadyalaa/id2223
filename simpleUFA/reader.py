import os

data_path="../data/0/"

def nextBatch(batch_size,batch_index):
	T_arr=[]
	q_arr=[]
	lwhr_arr=[]
	for i in range(batch_index*batch_size,batch_index*batch_size+batch_size):
		f = open(os.path.join(data_path,str(i)+".csv"))
		lines = f.readlines()
		T=[]
		q=[]
		lwhr=[]
		for j in range(1,len(lines)):
			items = lines[j].strip().split(",")
			T.append(float(items[2]))
			q.append(float(items[3]))
			lwhr.append(float(items[4]))
		f.close()
		T_arr.append(T)
		q_arr.append(q)
		lwhr_arr.append(lwhr)
	return T_arr, q_arr, lwhr_arr

def validBatch():
        T_arr=[]
        q_arr=[]
        lwhr_arr=[]
        for i in range(90000,91000):
                f = open(os.path.join(data_path,str(i)+".csv"))
                lines = f.readlines()
                T=[]
                q=[]
                lwhr=[]
                for j in range(1,len(lines)):
                        items = lines[j].strip().split(",")
                        T.append(float(items[2]))
                        q.append(float(items[3]))
                        lwhr.append(float(items[4]))
                f.close()
                T_arr.append(T)
                q_arr.append(q)
                lwhr_arr.append(lwhr)
        return T_arr, q_arr, lwhr_arr
