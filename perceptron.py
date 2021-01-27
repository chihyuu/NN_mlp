import tkinter as tk
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FuncFormatter
from tkinter import filedialog
from tkinter import ttk
import math

def do_openfile():
    filename = filedialog.askopenfilename()
    testfile_var.set(filename)

def readtext(filename):
    data = []
    with open(filename, 'r') as f:#with語句自動呼叫close()方法
        line = f.readline()
        while line:
            eachline = line.split()###按行讀取文字檔案，每行資料以列表形式返回
            read_data = [ float(x) for x in eachline[0:len(eachline)-1] ] #TopN概率字元轉換為float型
            lable = [ int(x) for x in eachline[-1] ]#lable轉換為int型
            read_data.append(lable[0])
            #read_data = list(map(float, eachline))
            data.append(read_data)
            line = f.readline()
        return data #返回資料為雙列表形式

def preprocess(data):
    label = [i[2] for i in data]
    ss = label[0]
    label[0] = 1
    data[0][2] = label[0]
    for i in range(1,len(data)):
        if label[i] == ss:
            label[i] = 1
        # else:
        #      label[i] = 1
        elif label[i] == ss+1 or label[i] == ss-1:
            label[i] = 0
        else:
            #data.remove(data[i][:])
            label[i] = 2
        data[i][2] = label[i]
    np.random.shuffle(data)
    train, test = np.array(data[:int(len(data)*2/3)]), np.array(data[int(len(data)*2/3):])
    #print(data)
    #print('train: ',train)
    #print('test: ',test)
    return train, test

def sig(value):
    if value>0.5:
        return 1
    else:
        return 0

def progress(currentValue):
    progressbar["value"]=currentValue

def perceptron(train, test, learning_rate, maxtime, initial_weight, theta, neuronum):
    #print(train)
    train_rate = 0
    test_rate = 0
    x_train = train[:, :train.shape[1]-1]
    x_train = np.insert(x_train, [0], -1, axis=1)
    y_train = train[:, -1]
    # print('xtrain',x_train,'ytrain',y_train)
    x_test = test[:, :test.shape[1]-1]
    x_test = np.insert(x_test, [0], -1, axis=1)
    y_test = test[:, -1]
    # print('xtest',x_test,'ytest',y_test)
    
    maxValue=maxtime
    currentValue=0
    progressbar["value"]=currentValue
    progressbar["maximum"]=maxValue-1

    weight = initial_weight
    same = False
    train_correct = 0
    train_error = 0
    test_correct = 0
    test_error = 0
    n= 0
    while not same and n < maxtime:
        progressbar["value"]=n
        progressbar.update()

        # print('\nRound: ',n)
        i = n % train.shape[0]
        y = [[] for k in range(neuronum)]
        for j in range(neuronum):
            y[j] = 1/(1+exp(-(np.dot(weight[j], x_train[i]))))
        # print('y=', y)
        z = 1/(1+exp(-(np.dot(weight[neuronum], (np.insert(y, [0], -1))))))
        # print('z=',z)
        predict = sig(z)
        # print(predict)
        s = [[] for k in range(neuronum+1)]
        w = [[] for k in range(neuronum+1)]
        if y_train[i]==predict:
            weight = weight
            train_correct += 1
            #same = True
        else:
            if predict == 1:
                s[neuronum] = (0-z)*z*(1-z)
                #s3 = (0-z)*z*(1-z)
            elif predict == 0:
                s[neuronum] = (1-z)*z*(1-z)
                #s3 = (1-z)*z*(1-z)
            for k in range (neuronum):
                s[k] = y[k]*(1-y[k])*s[neuronum]*weight[neuronum][k+1]
            #s1 = y1*(1-y1)*s3*weight[2][1]
            #s2 = y2*(1-y2)*s3*weight[2][2]
            #print('s1,s2,s3=',s1,s2,s3)
            # print('s=',s)
            for k in range (neuronum):
                w[k] = weight[k] + learning_rate*s[k]*x_train[i]
            # w0 = weight[0] + learning_rate*s[0]*x_train[i]
            # w1 = weight[1] + learning_rate*s[1]*x_train[i]
            w[neuronum] = weight[neuronum] + learning_rate*s[neuronum]*(np.insert(y, [0], -1))
            #w2 = weight[2] + learning_rate*s[2]*np.array([-1,y1,y2])
            weight = w
            train_error += 1
        # print("Weight:", weight)
        n += 1
    train_rate = train_correct*100/(train_error+train_correct)
    # print('train accuracy', train_correct*100/(train_error+train_correct), '%')
    #print("Weight:", weight)
    rmse = 0 

   
    for i in range(test.shape[0]):
        y = [[] for k in range(neuronum)]
        for j in range(neuronum):
            y[j] = 1/(1+exp(-(np.dot(weight[j], x_test[i]))))
        # y1 = 1/(1+exp(-(np.dot(weight[0], x_test[i]))))
        # y2 = 1/(1+exp(-(np.dot(weight[1], x_test[i]))))
        # z = 1/(1+exp(-(np.dot(weight[2], np.array([-1,y1,y2])))))
        z = 1/(1+exp(-(np.dot(weight[neuronum], (np.insert(y, [0], -1))))))
        predict = sig(z)
        # print(y,z)
        # print('\npredict: ',predict)
        # print('ans: ',y_test[i])
        rmse += (predict-y_test[i])** 2
        if y_test[i]!=predict:
            test_error += 1
        else:
            test_correct += 1
    rmse = (rmse/test.shape[0])**0.5
    # print('RMSE=', rmse)
    test_rate = test_correct*100/(test_error+test_correct)
    # print('test accuracy', test_correct*100/(test_error+test_correct), '%')

    #######transfornation
    new_train_x = [[] for i in range(x_train.shape[0])]
    new_train_y = [[] for i in range(x_train.shape[0])]
    new_z1 = [[] for i in range(x_train.shape[0])]
    new_test_x = [[] for i in range(x_test.shape[0])]
    new_test_y = [[] for i in range(x_test.shape[0])]
    new_z2 = [[] for i in range(x_test.shape[0])]
    for i in range(x_train.shape[0]):
        new_train_x[i] = 1/(1+exp(-(np.dot(weight[0], x_train[i]))))
        new_train_y[i] = 1/(1+exp(-(np.dot(weight[1], x_train[i]))))
        new_z1[i] = 1/(1+exp(-(np.dot(weight[2], [-1,new_train_x[i],new_train_y[i]]))))
    for i in range(x_test.shape[0]):
        new_test_x[i] = 1/(1+exp(-(np.dot(weight[0], x_test[i]))))
        new_test_y[i] = 1/(1+exp(-(np.dot(weight[1], x_test[i]))))
        new_z2[i] = 1/(1+exp(-(np.dot(weight[2], [-1,new_test_x[i],new_test_y[i]]))))

    new_train =np.array(list(zip(new_train_x, new_train_y, y_train, new_z1)))
    new_test = np.array(list(zip(new_test_x, new_test_y, y_test, new_z2)))

    train =np.array(list(zip(x_train[:, 1], x_train[:, 2], y_train, new_z1)))
    test = np.array(list(zip(x_test[:, 1], x_test[:, 2], y_test, new_z2)))
    # print('newtrain',new_train)
    # print('train',train)
    # print('test',new_test)
    plotData2D(train, test, weight, 1)
    if neuronum==2:
        plotData2D(new_train, new_test, weight, 2)
    plotData2D(train, test, weight, 3)
    #plotData3D(train, test, weight)
    return train_rate, test_rate, weight, rmse     

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plotData2D(train, test, w, num):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if num==1:
        ax.set_title('original data set')
    elif num==2:
        ax.set_title('transformation')
    else:
        ax.set_title('contourf')    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # print('#',train[:,:2])
    # print('###',train[:,2])
    # print('#####',test[:,:2])
    # print('#######',test[:,2])
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()  
    if num==1:
        canvas.get_tk_widget().place(x = 500, y = 10)
        toolbar = NavigationToolbar2Tk( canvas, window )
        toolbar.update()
        toolbar.place(x=700,y=0)
    elif num==2:
        x = np.arange(np.min(train) - 1, np.max(train) + 1, 0.1)
        if w[2][2]==0:
            y = 0
        else:
            y = (w[2][0] - w[2][1]*x) / w[2][2]
        ax.add_line(Line2D(x,y))
        canvas.get_tk_widget().place(x = 1150, y = 10)
        toolbar = NavigationToolbar2Tk( canvas, window )
        toolbar.update()
        toolbar.place(x=1300,y=0)
    else:
        X0, X1 = train[:,0], train[:,1]
        xx, yy = make_meshgrid(X0, X1)
        # print(xx.shape)
        # print(yy.shape)
        z = np.ones((xx.shape[0], xx.shape[1]))
        Z = z.reshape(xx.shape)
        #print(Z)
        new_xx = np.ones((xx.shape[0], xx.shape[1]))
        new_yy = np.ones((xx.shape[0], xx.shape[1]))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # print(xx[i][j],'!!!!',yy[i][j])
                new_xx[i][j] = 1/(1+exp(-(np.dot(w[0], [-1,xx[i][j],yy[i][j]]))))
                new_yy[i][j] = 1/(1+exp(-(np.dot(w[1], [-1,xx[i][j],yy[i][j]]))))
                #print(new_xx[i][j],'newy',new_yy[i][j])
                Z[i][j] = 1/(1+exp(-(np.dot(w[2], [-1,new_xx[i][j],new_yy[i][j]]))))
        # print('z',Z)
        # print('xx=', xx, 'yy=', yy)
        # print('Z=',Z)
        cm = plt.cm.RdBu
        a = alpa_var.get()
        ax.contourf(xx, yy, Z, cmap=cm, alpha=a)
        #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        canvas.get_tk_widget().place(x = 500, y = 500)
        toolbar = NavigationToolbar2Tk( canvas, window )
        toolbar.update()
        toolbar.place(x=700,y=490)
    labels = array(train[:,2])
    idx_1 = where(train[:,2]==1)
    p1 = ax.scatter(train[idx_1,0], train[idx_1,1], 
        marker='o', color='g', label='train-0', s=20)
    idx_2 = where(train[:,2]==0)
    p2 = ax.scatter(train[idx_2,0], train[idx_2,1], 
        marker='x', color='blue', label='train-1', s=20)
    
    idx_3 = where(test[:,2]==1)
    p3 = ax.scatter(test[idx_3,0], test[idx_3,1],
        marker='o', color='limegreen', label='test-0', s=20)
    idx_4 = where(test[:,2]==0)
    p4 = ax.scatter(test[idx_4,0], test[idx_4,1],
        marker='x', color='deepskyblue', label='test-1', s=20)

    idx_5 = where(train[:,2]==2)
    p5 = ax.scatter(train[idx_5,0], train[idx_5,1], 
        marker='.', color='black', s=20)
    idx_6 = where(test[:,2]==2)
    p6 = ax.scatter(test[idx_6,0], test[idx_6,1], 
        marker='.', color='black', s=20)
    plt.legend(loc = 'upper right')

def plotData3D(train, test, w):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title('3D')
    plt.xlabel('X')
    plt.ylabel('Y')
    X0, X1 = train[:,0], train[:,1]
    X, Y = make_meshgrid(X0, X1)
    Z = (-w[2][0]*X - w[2][1]*Y) / w[2][2]
    surf = ax.plot_surface(X,Y,Z)

    idx_1 = where(train[:,2]==1)
    p1 = ax.scatter(train[idx_1,0], train[idx_1,1], train[idx_1,3],
        marker='o', color='g', label='train-0', s=20)
    idx_2 = where(train[:,2]==0)
    p2 = ax.scatter(train[idx_2,0], train[idx_2,1], train[idx_2,3], 
        marker='x', color='black', label='train-1', s=20)
    
    idx_3 = where(test[:,2]==1)
    p3 = ax.scatter(test[idx_3,0], test[idx_3,1], test[idx_3,3], 
        marker='o', color='limegreen', label='test-0', s=20)
    idx_4 = where(test[:,2]==0)
    p4 = ax.scatter(test[idx_4,0], test[idx_4,1], test[idx_4,3],
        marker='x', color='darkgray', label='test-1', s=20)

    idx_5 = where(train[:,2]==2)
    p5 = ax.scatter(train[idx_5,0], train[idx_5,1], train[idx_5,3],
        marker='.', color='snow', s=20)
    idx_6 = where(test[:,2]==2)
    p6 = ax.scatter(test[idx_6,0], test[idx_6,1], test[idx_6,3],
        marker='.', color='snow', s=20) 

    plt.legend(loc = 'upper right')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()  
    canvas.get_tk_widget().place(x = 1100, y = 500)
    toolbar = NavigationToolbar2Tk( canvas, window )
    toolbar.update()
    toolbar.place(x=1300,y=490)

def enter():
    '''for widget in frame.winfo_children():
       widget.destroy()'''
    done_frame = tk.Canvas(window)
    done_frame.place(x = 10, y = 900)
    filename = str(testfile_var.get())
    learning_rate = float(learning_rate_var.get())
    maxtime = float(convergence_var.get())
    neuronum = int(neuronum_var.get())
    if neuronum!= 2:
        frame = tk.Canvas(window, height=1000, width=700)
        frame.place(x = 1150, y = 0)
    theta = -1
    data = readtext(filename)
    train, test = preprocess(data)
    # label_var.set(np.round(train,4))
    # label.place(x=1000,y=10)
    #s1 = tk.Scrollbar(text_win,orient=tk.VERTICAL)
    
    # train_label = tk.Label(text_win, text='train set').place(x = 1000, y = 50)
    # for i in range(len(train)):
    #     train_listbox.insert(tk.END, train[i])
    # train_listbox.place(x = 1000, y = 80)
    # train_scrollbar.config(command=train_listbox.yview)

    # test_label = tk.Label(text_win, text='test set').place(x = 1000, y = 300)
    # for i in range(len(test)):
    #     test_listbox.insert(tk.END, test[i])
    # test_listbox.place(x = 1000, y = 330)
    # test_scrollbar.config(command=test_listbox.yview)

    # print(train.shape[0],train.shape[1])
    initial_weight = [[] for i in range(neuronum+1)]
    for i in range(neuronum):
        initial_weight[i] = np.round(np.random.rand(train.shape[1]),2)
    initial_weight[neuronum] = np.round(np.random.rand(neuronum+1),2)
    #initial_weight = [-1.2,1,1],[0.3,1,1],[0.5,0.4,0.8]
    train_rate, test_rate, weight, rmse = perceptron(train, test, learning_rate, maxtime, initial_weight, theta, neuronum)
    result_w = [[] for i in range(neuronum+1)]
    result1 = '訓練辨識率：{} %'.format(round(train_rate,4))
    result2 = '測試辨識率：{} %'.format(round(test_rate,4))
    result3 = 'RMSE：{}'.format(np.round(rmse,4))
     
    for i in range(neuronum+1):
        result_w[i] = '收斂後鍵結值{} : {}              '.format(i,np.round(weight[i],4))    
    
    output_frame = tk.Canvas(window, height=400, width=300)
    output_frame.place(x = 0, y = 270)
    result1_label = tk.Label(output_frame, text=result1).place(x = 10, y = 0)
    result2_label = tk.Label(output_frame, text=result2).place(x = 10, y = 30)
    result3_label = tk.Label(output_frame, text=result3).place(x = 10, y = 60)
    for i in range(neuronum+1):
        result_w_label = tk.Label(output_frame, text=result_w[i]).place(x = 10, y = 100+(i*20))
    output_frame.delete()
    tk.Label(done_frame, text='DONE', bd=10, bg='salmon').place(x = 0, y = 0)
    
if __name__ == '__main__':

    window = tk.Tk()
    window.title('perceptron')
    window.geometry('1800x1000')
    #window.configure(background='white')
    
    #text_win = tk.Frame(window)

    header_label = tk.Label(window, text='多層感知機類神經網路').place(x = 10, y = 0)
    
    testfile_var = tk.StringVar()
    file_btn = tk.Button(window, text="openfile", command=do_openfile).place(x = 100, y = 20)
    
    
    testfile_label = tk.Label(window, text='測資').place(x = 10, y = 50)
    testfile_entry = tk.Entry(window, textvariable=testfile_var, width=70).place(x = 100 , y = 50)
    # testfile_combo = ttk.Combobox(window, textvariable=testfile_var, 
    #                                 values=[
    #                                 "2Ccircle1.txt", 
    #                                 "2Circle1.txt",
    #                                 "2Circle2.txt",
    #                                 "2CloseS.txt",
    #                                 "2CloseS2.txt",
    #                                 "2CloseS3.txt",
    #                                 "2cring.txt",
    #                                 "2CS.txt",
    #                                 "2Hcircle1.txt",
    #                                 "2ring.txt",
    #                                 "perceptron1.txt",
    #                                 "perceptron2.txt"]).place(x = 100, y = 50)

    learning_rate_var = tk.DoubleVar()
    learning_rate_var.set(0.5)
    learning_rate_label = tk.Label(window, text='學習率').place(x = 10, y = 80)
    learning_rate_entry = tk.Entry(window, textvariable=learning_rate_var).place(x = 100 , y = 80)

    convergence_var = tk.DoubleVar()
    convergence_var.set(10000)
    convergence_label = tk.Label(window, text='收斂條件(次數)').place(x = 10, y = 110)
    convergence_entry = tk.Entry(window, textvariable=convergence_var).place(x = 100, y = 110)
    
    neuronum_var = tk.IntVar()
    neuronum_var.set(2)
    neuronum_label = tk.Label(window, text='神經元個數').place(x = 10, y = 140)
    neuronum_entry = tk.Entry(window, textvariable=neuronum_var).place(x = 100, y = 140)

    alpa_var = tk.DoubleVar()
    alpa_var.set(0.8)
    alpa_label = tk.Label(window, text='等高線圖alpa值').place(x = 10, y = 170)
    alpa_entry = tk.Entry(window, textvariable=alpa_var).place(x = 100, y = 170)
    s = ttk.Style() 
    s.theme_use('clam') 
    s.configure("red.Horizontal.TProgressbar", foreground='salmon', background='salmon') 
    bar_label = tk.Label(window, text='訓練進度條').place(x = 10, y = 800)
    progressbar=ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate", style="red.Horizontal.TProgressbar")
    progressbar.place(x = 10, y = 820)
    #text = tk.Text(window, height=3).place(x = 1000, y = 10)
    # train_scrollbar = tk.Scrollbar(window)    
    # label_var = tk.StringVar()
    # label = tk.Label(window, textvariable=label_var)

    # train_scrollbar = tk.Scrollbar(text_win)    
    # train_listbox = tk.Listbox(text_win, yscrollcommand=train_scrollbar.set)

    # test_scrollbar = tk.Scrollbar(text_win)    
    # test_listbox = tk.Listbox(text_win, yscrollcommand=test_scrollbar.set)

    # s_var = tk.IntVar()
    # s = tk.Scale(window, label='zoomer(數字越大縮越小)', from_=1, to=3, orient=tk.HORIZONTAL, variable=s_var,
    #             length=200, showvalue=1, tickinterval=1, resolution=0.01).place(x = 10, y = 800)

    calculate_btn = tk.Button(window, text='顯示訓練結果', command=enter).place(x = 10, y = 200)    

    window.mainloop()

