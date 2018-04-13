X = []
Y= []
with open('CAT1.txt', 'r') as f_train:
	with open('CAT2.txt','w')as r:
     		lines=f_train.readlines()
     		for line in lines:
        		my_list_train = line.split(" ") 
        #print my_list_train   # replace with your own separator instead
			r.write(str(my_list_train[:-1]))
			r.write("\n") # omitting identifier in [0] and target in [-1]
        	#print str(line)
        		Y.append(my_list_train[-1])
#print X
f_train.close() 
r.close()
'''     	
with open('CAT2.txt','w')as r:
	#r.write(X)
	for i in (0,len(X)-1):
		print(X[i])
		j=X[i]
		r.write(str(j))
		r.write('\n') 
		
r.close()		       	
 '''      	
        	
